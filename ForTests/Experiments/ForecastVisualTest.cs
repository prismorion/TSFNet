using TSFNet.Calculations;
using TSFNet.Data;
using TSFNet.Models.GRU;
using TSFNet.Models.MLP;
using TSFNet.Models.RNN;
using TSFNet.Training;
using TSFNet.Training.Parameters;
using ScottPlot;
using ScottPlot.WinForms;

namespace ForTests.Experiments
{
    public class ForecastVisualTest
    {
        public static void Run(string csvPath, int windowSize, int hiddenSize, int seed = 0)
        {
            double trainProportion = 0.7;
            double valProportion = 0.15;

            List<Dot> records = DotCSVHelper.Read(csvPath);
            var allY = records.Select(r => r.Y).ToArray();

            int trainDots = (int)(allY.Length * trainProportion);

            // нормализация z-score по трейну
            var scaler = new StandardScaler(allY.Take(trainDots).ToArray());
            var scaledY = scaler.Transform(allY);

            // приведение к стационарности
            double[] diffScaledY = Differencer.Diff(scaledY);

            var preparedRecords = new List<Dot>();
            for (int i = 0; i < diffScaledY.Length; i++)
                preparedRecords.Add(new Dot(records[i + 1].X, diffScaledY[i]));

            Hyperparameters hyperparameters = new Hyperparameters();
            hyperparameters.learningRate = 0.01;
            hyperparameters.batchSize = 8;
            hyperparameters.l2Lambda = 0;
            hyperparameters.threshold = 5;

            TrainingOptions trainingOptions = new TrainingOptions();
            trainingOptions.epochs = 2000;
            trainingOptions.reportEvery = 0;
            trainingOptions.patience = 200;

            // подготовка данных
            var inputs = DataPreparator.InputsPreparation(preparedRecords, windowSize, 1);
            var seqInputs = DataPreparator.SeqInputsPreparation(preparedRecords, windowSize, 1);
            var targets = DataPreparator.TargetsPreparation(preparedRecords, windowSize, 1);

            int samples = inputs.Length;
            int trainSize = (int)(samples * trainProportion);
            int valSize = (int)(samples * valProportion);
            int testSize = samples - trainSize - valSize;
            int offset = trainSize + valSize;

            // обучающие
            var inpTrain = inputs.Take(trainSize).ToArray();
            var seqInpTrain = seqInputs.Take(trainSize).ToArray();
            var outTrain = targets.Take(trainSize).ToArray();

            // валидационные
            var inpVal = inputs.Skip(trainSize).Take(valSize).ToArray();
            var seqInpVal = seqInputs.Skip(trainSize).Take(valSize).ToArray();
            var outVal = targets.Skip(trainSize).Take(valSize).ToArray();

            Dataset<double[]> trainDataset = new Dataset<double[]>(inpTrain, outTrain);
            Dataset<double[][]> trainSeqDataset = new Dataset<double[][]>(seqInpTrain, outTrain);
            Dataset<double[]> valDataset = new Dataset<double[]>(inpVal, outVal);
            Dataset<double[][]> valSeqDataset = new Dataset<double[][]>(seqInpVal, outVal);

            // обучение
            RandGen.Seed(seed);

            MLP mlp = new MLP(windowSize, hiddenSize, 1);
            RNN rnn = new RNN(1, hiddenSize, 1);
            GRU gru = new GRU(1, hiddenSize, 1);

            Trainer.FitEarlyStopping(mlp, trainDataset, valDataset, hyperparameters, trainingOptions);
            Trainer.FitEarlyStopping(rnn, trainSeqDataset, valSeqDataset, hyperparameters, trainingOptions);
            Trainer.FitEarlyStopping(gru, trainSeqDataset, valSeqDataset, hyperparameters, trainingOptions);

            // подготовка массивов для хранения предсказаний
            double[] xsTest = new double[testSize];
            double[] yActual = new double[testSize];
            double[] yMlp = new double[testSize];
            double[] yRnn = new double[testSize];
            double[] yGru = new double[testSize];

            // стартовые окна - последние истинные диффы перед тестом
            double[] windowMlp = (double[])inputs[offset].Clone();
            double[][] windowRnn = CloneSeq(seqInputs[offset]);
            double[][] windowGru = CloneSeq(seqInputs[offset]);

            double anchor = scaledY[offset + windowSize];
            double mlpDiffSum = 0, rnnDiffSum = 0, gruDiffSum = 0;

            // авторегрессия
            for (int k = 0; k < testSize; k++)
            {
                double mlpDiff = mlp.Predict(windowMlp)[0];
                double rnnDiff = rnn.Predict(windowRnn)[0];
                double gruDiff = gru.Predict(windowGru)[0];

                mlpDiffSum += mlpDiff;
                rnnDiffSum += rnnDiff;
                gruDiffSum += gruDiff;

                int idx = offset + windowSize + 1 + k;
                xsTest[k] = records[idx].X;
                yActual[k] = allY[idx];
                yMlp[k] = scaler.InverseTransform(anchor + mlpDiffSum);
                yRnn[k] = scaler.InverseTransform(anchor + rnnDiffSum);
                yGru[k] = scaler.InverseTransform(anchor + gruDiffSum);

                RefreshWindow(windowMlp, mlpDiff);
                RefreshWindow(windowRnn, rnnDiff);
                RefreshWindow(windowGru, gruDiff);
            }

            // вывод результатов
            ShowMetrics(yActual, yMlp, yRnn, yGru);
            Plot("Авторегрессионный прогноз на тестовом участке", xsTest, yActual, yMlp, yRnn, yGru);
        }

        private static void ShowMetrics(double[] actual, double[] mlp, double[] rnn, double[] gru)
        {
            double[][] preds = { mlp, rnn, gru };

            Console.WriteLine("Метрики авторегрессионного прогноза на тесте");
            Console.WriteLine("|" + new string('-', 48) + "|");
            Console.WriteLine($"|{"",8} | {"MLP",10} | {"RNN",10} | {"GRU",10} |");
            Console.WriteLine("|" + new string('-', 48) + "|");

            PrintRow("MSE", preds, p => Metrics.MSE(actual, p));
            PrintRow("RMSE", preds, p => Metrics.RMSE(actual, p));
            PrintRow("MAE", preds, p => Metrics.MAE(actual, p));
            PrintRow("R^2", preds, p => R2(actual, p));
            Console.WriteLine("|" + new string('-', 48) + "|");
        }

        private static void PrintRow(string metric, double[][] preds, Func<double[], double> f)
        {
            Console.WriteLine($"|{metric,8} | {f(preds[0]),10:F4} | {f(preds[1]),10:F4} | {f(preds[2]),10:F4} |");
        }

        private static double R2(double[] yTrue, double[] yPred)
        {
            double mean = 0;
            for (int i = 0; i < yTrue.Length; i++)
                mean += yTrue[i];
            mean /= yTrue.Length;

            double ssRes = 0, ssTot = 0;
            for (int i = 0; i < yTrue.Length; i++)
            {
                double res = yTrue[i] - yPred[i];
                double tot = yTrue[i] - mean;
                ssRes += res * res;
                ssTot += tot * tot;
            }
            return 1.0 - ssRes / ssTot;
        }

        private static void Plot(string title, double[] xs, double[] actual,
            double[] mlp, double[] rnn, double[] gru)
        {
            var plt = new Plot();

            var act = plt.Add.ScatterLine(xs, actual);
            act.LegendText = "Факт";
            act.Color = Colors.Black;
            act.LineWidth = 2;

            var mlpLine = plt.Add.ScatterLine(xs, mlp);
            mlpLine.LegendText = "MLP";
            mlpLine.Color = Colors.Green;
            mlpLine.LineWidth = 2;

            var rnnLine = plt.Add.ScatterLine(xs, rnn);
            rnnLine.LegendText = "RNN";
            rnnLine.Color = Colors.Red;
            rnnLine.LineWidth = 2;

            var gruLine = plt.Add.ScatterLine(xs, gru);
            gruLine.LegendText = "GRU";
            gruLine.Color = Colors.Blue;
            gruLine.LineWidth = 2;

            plt.Title(title);
            plt.XLabel("X");
            plt.YLabel("Y");
            plt.ShowLegend(Edge.Right);

            FormsPlotViewer.Launch(plt, "Исследование эффективности нейронных сетей");
        }

        private static void RefreshWindow(double[] window, double num)
        {
            for (int i = 0; i < window.Length - 1; i++)
                window[i] = window[i + 1];
            window[^1] = num;
        }

        private static void RefreshWindow(double[][] window, double num)
        {
            for (int i = 0; i < window.Length - 1; i++)
                window[i] = window[i + 1];
            window[^1] = new double[] { num };
        }

        private static double[][] CloneSeq(double[][] seq)
        {
            var copy = new double[seq.Length][];
            for (int i = 0; i < seq.Length; i++)
                copy[i] = (double[])seq[i].Clone();
            return copy;
        }
    }
}