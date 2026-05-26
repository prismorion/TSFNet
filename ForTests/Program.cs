using TSFNet.Training;
using TSFNet.Models.RNN;
using TSFNet.Models.MLP;
using TSFNet.Models.GRU;
using TSFNet.Data;
using TSFNet.Training.Parameters;
using TSFNet.Training.Responses;
using TSFNet.Calculations;
using ScottPlot;
using ScottPlot.WinForms;

namespace ForTests
{
    class Programm
    {
        static void Main(string[] args)
        {
            double trainProportion = 0.7;

            // === Чтение CSV ===
            string csvPath = "sin_cos_trend_bias.csv";
            List<Dot> records = DotCSVHelper.Read(csvPath);

            // === Препроцессинг ===
            var allY = records.Select(r => r.Y).ToArray();
            int trainDots = (int)(allY.Length * trainProportion);

            var scaler = new StandardScaler(allY.Take(trainDots).ToArray());
            var scaledY = scaler.Transform(allY);

            var preparedRecords = new List<Dot>();
            for (int i = 0; i < scaledY.Length; i++)
                preparedRecords.Add(new Dot(records[i].X, scaledY[i]));

            // === Подготовка данных ===
            int windowSize = 10;
            int output = 1;
            var inputs = DataPreparator.InputsPreparation(preparedRecords, windowSize, output);
            var seqInputs = DataPreparator.SeqInputsPreparation(preparedRecords, windowSize, output);
            var targets = DataPreparator.TargetsPreparation(preparedRecords, windowSize, output);

            // === Сплит ===
            int samples = (preparedRecords.Count - windowSize) / output;
            int trainSize = (int)(samples * trainProportion);
            int testSize = samples - trainSize;
            // данные для обучения
            var inpTrain = inputs.Take(trainSize).ToArray();
            var outTrain = targets.Take(trainSize).ToArray();
            // данные для тестов
            var inpTest = inputs.Skip(trainSize).ToArray();
            var outTest = targets.Skip(trainSize).ToArray();
            var seqInpTrain = seqInputs.Take(trainSize).ToArray();
            var seqInpTest = seqInputs.Skip(trainSize).ToArray();

            // === Подготовка датасетов ===
            Dataset<double[]> trainDataset = new Dataset<double[]>(inpTrain, outTrain);
            Dataset<double[][]> trainSeqDataset = new Dataset<double[][]>(seqInpTrain, outTrain);
            Dataset<double[]> testDataset = new Dataset<double[]>(inpTest, outTest);
            Dataset<double[][]> testSeqDataset = new Dataset<double[][]>(seqInpTest, outTest);

            // === Модели нейронных сетей ===
            int hidden = 32;
            MLP mlp = new MLP([windowSize, hidden, 16, output]);
            RNN rnn = new RNN(1, hidden, output);
            GRU gru = new GRU(1, hidden, output);

            // === Гиперпараметры обучения ===
            Hyperparameters hyperparameters = new Hyperparameters();
            hyperparameters.learningRate = 0.01;
            hyperparameters.batchSize = 2;
            hyperparameters.l2Lambda = 0.01;
            hyperparameters.threshold = 5;

            TrainingOptions trainingOptions = new TrainingOptions();
            trainingOptions.epochs = 100;
            trainingOptions.reportEvery = 10;

            // === Обучение моделей и вывод результата ===
            FitResponse MLPResponse = Trainer.Fit(mlp, trainDataset, hyperparameters, trainingOptions);
            Console.WriteLine("=== MLP ===");
            ResponsePrinter.Print(MLPResponse);
            Console.WriteLine();

            hyperparameters.l2Lambda = 0;

            FitResponse RNNResponse = Trainer.Fit(rnn, trainSeqDataset, hyperparameters, trainingOptions);
            Console.WriteLine("=== RNN ===");
            ResponsePrinter.Print(RNNResponse);
            Console.WriteLine();
            
            FitResponse GRUResponse = Trainer.Fit(gru, trainSeqDataset, hyperparameters, trainingOptions);
            Console.WriteLine("=== GRU ===");
            ResponsePrinter.Print(GRUResponse);
            Console.WriteLine();

            Console.WriteLine("\n=== Метрики ===");
            PrintMetrics("MLP", testDataset, mlp.Forward);
            PrintMetrics("RNN", testSeqDataset, rnn.Forward);
            PrintMetrics("GRU", testSeqDataset, gru.Forward);

            // === Построение графика ===
            string name = "y = 0.2x + 2sin(0.2x) + 0.5cos(x) - cos(0.7x) + 2";
            // исходный график
            var xs = new double[records.Count];
            var ys = new double[records.Count];
            for (int i = 0; i < records.Count; i++)
            {
                xs[i] = records[i].X;
                ys[i] = records[i].Y;
            }

            // предсказание моделей
            Dataset<double[]> inpDataset = new Dataset<double[]>(inputs, targets);
            Dataset<double[][]> inpSeqDataset = new Dataset<double[][]>(seqInputs, targets);

            int totalPoints = samples * output;
            var xsPred = new double[totalPoints];
            for (int k = 0; k < totalPoints; k++)
                xsPred[k] = records[windowSize + k].X;

            var predMLP = PredictTeacherForced(mlp.Forward, inputs, scaler, records, windowSize, output);
            var predRNN = PredictTeacherForced(rnn.Forward, seqInputs, scaler, records, windowSize, output);
            var predGRU = PredictTeacherForced(gru.Forward, seqInputs, scaler, records, windowSize, output);

            ModelPlot mlpPlot = new ModelPlot("MLP", System.Drawing.Color.Green, xsPred, predMLP);
            ModelPlot rnnPlot = new ModelPlot("RNN", System.Drawing.Color.Red, xsPred, predRNN);
            ModelPlot gruPlot = new ModelPlot("GRU", System.Drawing.Color.Blue, xsPred, predGRU);

            PlotGraph(name, xs, ys, trainProportion, mlpPlot, rnnPlot, gruPlot);
        }

        public static double[] PredictTeacherForced<TInput>(Func<TInput, double[]> forward, TInput[] inputs,
            StandardScaler scaler, List<Dot> records, int windowSize, int output)
        {
            int samples = inputs.Length;
            var result = new double[samples * output];

            for (int i = 0; i < samples; i++)
            {
                var p = forward(inputs[i]);
                for (int j = 0; j < output; j++)
                {
                    int k = i * output + j;
                    result[k] = scaler.InverseTransform(p[j]);
                }
            }

            return result;
        }

        public static void PrintMetrics<TInput>(string model, Dataset<TInput> dataset, Func<TInput, double[]> predict)
        {
            double mse = Metrics.MSE(dataset, predict);
            double rmse = Metrics.RMSE(dataset, predict);
            double mae = Metrics.MAE(dataset, predict);
            Console.WriteLine($"{model}\tMSE = {mse:F6}\tRMSE = {rmse:F6}\tMAE = {mae:F6}");
        }

        [STAThread]
        public static void PlotGraph(string name, double[] xs, double[] ys, double trainSize, params ModelPlot[] modelPlot)
        {
            var plt = new Plot();

            int splitIdx = (int)(xs.Length * trainSize);
            double trainEndX = xs[splitIdx];

            // Train
            var trainSpan = plt.Add.HorizontalSpan(xs[0], trainEndX);
            trainSpan.FillStyle.Color = Colors.LightGreen.WithAlpha(0.35);
            trainSpan.LineStyle.Width = 0;
            trainSpan.LegendText = "Train";

            // Test
            var testSpan = plt.Add.HorizontalSpan(trainEndX, xs[^1]);
            testSpan.FillStyle.Color = Colors.LightSeaGreen.WithAlpha(0.25);
            testSpan.LineStyle.Width = 0;
            testSpan.LegendText = "Test";

            var actual = plt.Add.ScatterLine(xs, ys);
            actual.LegendText = "Actual";
            actual.Color = Colors.Black;
            actual.LineWidth = 2;

            foreach (var model in modelPlot)
            {
                var pModel = plt.Add.ScatterLine(model.X, model.Y);
                pModel.LegendText = model.Name;
                pModel.Color = ScottPlot.Color.FromColor(model.Color);
                pModel.LineWidth = 2;
            }

            plt.Title(name);
            plt.XLabel("X");
            plt.YLabel("Y");
            plt.ShowLegend();

            plt.ShowLegend(Edge.Right);
            FormsPlotViewer.Launch(plt, "Исследование эффективности нейронных сетей");
        }
    }
}
