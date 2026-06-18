using TSFNet.Data;
using TSFNet.Models.GRU;
using TSFNet.Models.MLP;
using TSFNet.Models.RNN;
using TSFNet.Training;
using TSFNet.Training.Parameters;
using TSFNet.Calculations;

namespace ForTests.Experiments
{
    public class WindowSizeExperiment
    {
        public static Dictionary<int, Dictionary<string, double>> Run(string csvPath, int[] windowSizes, int hiddenSize, int repeats = 5)
        {
            double trainProportion = 0.7;
            double valProportion = 0.15;

            var result = new Dictionary<int, Dictionary<string, double>>();

            List<Dot> records = DotCSVHelper.Read(csvPath);
            var allY = records.Select(r => r.Y).ToArray();

            int trainDots = (int)(allY.Length * trainProportion);

            // нормализация z-score по трейну
            var scaler = new StandardScaler(allY.Take(trainDots).ToArray());
            var scaledY = scaler.Transform(allY);

            // приведение к стационарности
            double[] diffScaledY = Differencer.Diff(scaledY);

            // при дифференцировании теряется первое значение
            var preparedRecords = new List<Dot>();
            for (int i = 0; i < diffScaledY.Length; i++)
                preparedRecords.Add(new Dot(records[i + 1].X, diffScaledY[i]));

            // масштаб MASE: наивная MAE на трейне исходного ряда
            double scale = 0;
            for (int k = 1; k < trainDots; k++)
                scale += Math.Abs(allY[k] - allY[k - 1]);
            scale /= (trainDots - 1);

            Hyperparameters hyperparameters = new Hyperparameters();
            hyperparameters.learningRate = 0.01;
            hyperparameters.batchSize = 4;
            hyperparameters.l2Lambda = 0;
            hyperparameters.threshold = 5;

            TrainingOptions trainingOptions = new TrainingOptions();
            trainingOptions.epochs = 1000;
            trainingOptions.reportEvery = 0;
            trainingOptions.patience = 150;

            for (int i = 0; i < windowSizes.Length; i++)
            {
                int w = windowSizes[i];

                var inputs = DataPreparator.InputsPreparation(preparedRecords, w, 1);
                var seqInputs = DataPreparator.SeqInputsPreparation(preparedRecords, w, 1);
                var targets = DataPreparator.TargetsPreparation(preparedRecords, w, 1);

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

                // тестовые
                var inpTest = inputs.Skip(offset).Take(testSize).ToArray();
                var seqInpTest = seqInputs.Skip(offset).Take(testSize).ToArray();

                Dataset<double[]> trainDataset = new Dataset<double[]>(inpTrain, outTrain);
                Dataset<double[][]> trainSeqDataset = new Dataset<double[][]>(seqInpTrain, outTrain);
                Dataset<double[]> valDataset = new Dataset<double[]>(inpVal, outVal);
                Dataset<double[][]> valSeqDataset = new Dataset<double[][]>(seqInpVal, outVal);

                double mlpMaeSum = 0, rnnMaeSum = 0, gruMaeSum = 0;

                for (int r = 0; r < repeats; r++)
                {
                    RandGen.Seed(r);

                    MLP mlp = new MLP(w, hiddenSize, 1);
                    RNN rnn = new RNN(1, hiddenSize, 1);
                    GRU gru = new GRU(1, hiddenSize, 1);

                    Trainer.FitEarlyStopping(mlp, trainDataset, valDataset, hyperparameters, trainingOptions);
                    Trainer.FitEarlyStopping(rnn, trainSeqDataset, valSeqDataset, hyperparameters, trainingOptions);
                    Trainer.FitEarlyStopping(gru, trainSeqDataset, valSeqDataset, hyperparameters, trainingOptions);

                    // прогноз по тесту
                    double[][] mlpPreds = new double[inpTest.Length][];
                    for (int j = 0; j < inpTest.Length; j++)
                        mlpPreds[j] = mlp.Predict(inpTest[j]);

                    double[][] rnnPreds = new double[seqInpTest.Length][];
                    double[][] gruPreds = new double[seqInpTest.Length][];
                    for (int j = 0; j < seqInpTest.Length; j++)
                    {
                        rnnPreds[j] = rnn.Predict(seqInpTest[j]);
                        gruPreds[j] = gru.Predict(seqInpTest[j]);
                    }

                    // восстановление в исходные единицы
                    double[] yTrue = new double[inpTest.Length];
                    double[] mlpRestored = new double[inpTest.Length];
                    double[] rnnRestored = new double[inpTest.Length];
                    double[] gruRestored = new double[inpTest.Length];

                    for (int j = 0; j < inpTest.Length; j++)
                    {
                        int s = offset + j;
                        double prevScaled = scaledY[s + w];

                        yTrue[j] = allY[s + w + 1];
                        mlpRestored[j] = scaler.InverseTransform(prevScaled + mlpPreds[j][0]);
                        rnnRestored[j] = scaler.InverseTransform(prevScaled + rnnPreds[j][0]);
                        gruRestored[j] = scaler.InverseTransform(prevScaled + gruPreds[j][0]);
                    }

                    mlpMaeSum += Metrics.MAE(yTrue, mlpRestored);
                    rnnMaeSum += Metrics.MAE(yTrue, rnnRestored);
                    gruMaeSum += Metrics.MAE(yTrue, gruRestored);
                }

                double mlpMae = mlpMaeSum / repeats;
                double rnnMae = rnnMaeSum / repeats;
                double gruMae = gruMaeSum / repeats;

                result[w] = new Dictionary<string, double>
                {
                    { "mlp", mlpMae / scale},
                    { "rnn", rnnMae / scale},
                    { "gru", gruMae / scale},
                    { "mlpMae", mlpMae},
                    { "rnnMae", rnnMae},
                    { "gruMae", gruMae},
                    { "naive", scale}
                };
            }

            return result;
        }

        public static void ShowResultWindowSizeExperiment(Dictionary<int, Dictionary<string, double>> result)
        {
            double naive = result.Values.First()["naive"];

            Console.WriteLine("MASE по размеру скользящего окна (меньше 1 = лучше наивного)");
            Console.WriteLine(new string('-', 47));
            Console.WriteLine($"{"Окно",8} | {"MLP",10} | {"RNN",10} | {"GRU",10}");
            Console.WriteLine(new string('-', 47));
            foreach (int key in result.Keys.OrderBy(k => k))
            {
                var r = result[key];
                Console.WriteLine($"{key,8} | {r["mlp"],10:F3} | {r["rnn"],10:F3} | {r["gru"],10:F3}");
            }
            Console.WriteLine(new string('-', 47));

            Console.WriteLine();

            Console.WriteLine($"Наивная MAE (baseline, повтор последнего значения): {naive:F4}");
            Console.WriteLine();

            Console.WriteLine("MAE моделей");
            Console.WriteLine(new string('-', 47));
            Console.WriteLine($"{"Окно",8} | {"MLP",10} | {"RNN",10} | {"GRU",10}");
            Console.WriteLine(new string('-', 47));
            foreach (int key in result.Keys.OrderBy(k => k))
            {
                var r = result[key];
                Console.WriteLine($"{key,8} | {r["mlpMae"],10:F4} | {r["rnnMae"],10:F4} | {r["gruMae"],10:F4}");
            }
            Console.WriteLine(new string('-', 47));
        }
    }
}