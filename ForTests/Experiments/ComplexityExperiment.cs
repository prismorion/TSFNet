using TSFNet.Calculations;
using TSFNet.Data;
using TSFNet.Models.GRU;
using TSFNet.Models.MLP;
using TSFNet.Models.RNN;
using TSFNet.Training;
using TSFNet.Training.Parameters;
using TSFNet.Training.Responses;

namespace ForTests.Experiments
{
    public static class ComplexityExperiment
    {
        public static Dictionary<int, Dictionary<string, double>> Run(string csvPath, int windowSize, int[] hiddenSizes, int repeats = 5)
        {
            Dictionary<int, Dictionary<string, double>> result = new Dictionary<int, Dictionary<string, double>> { };

            // считывание и разделение данных
            List<Dot> records = DotCSVHelper.Read(csvPath);

            var inputs = DataPreparator.InputsPreparation(records, windowSize, 1);
            var seqInputs = DataPreparator.SeqInputsPreparation(records, windowSize, 1);
            var targets = DataPreparator.TargetsPreparation(records, windowSize, 1);

            Dataset<double[]> trainDataset = new Dataset<double[]>(inputs, targets);
            Dataset<double[][]> trainSeqDataset = new Dataset<double[][]>(seqInputs, targets);

            // гиперпараметры и параметры обучения
            Hyperparameters hyperparameters = new Hyperparameters();
            hyperparameters.learningRate = 0.01;
            hyperparameters.batchSize = 1;
            hyperparameters.threshold = 5;

            TrainingOptions trainingOptions = new TrainingOptions();
            trainingOptions.epochs = 200;

            // прогрев JIT
            Calculation(windowSize, hiddenSizes[0], trainDataset, trainSeqDataset, hyperparameters, trainingOptions);            

            double[] mlpResult = new double[repeats];
            double[] rnnResult = new double[repeats];
            double[] gruResult = new double[repeats];

            for (int i = 0; i < hiddenSizes.Length; i++)
            {
                for(int r = 0; r < repeats; r++)
                {
                    RandGen.Seed(r);
                    Dictionary<string, FitResponse> calculationResult = Calculation(windowSize, hiddenSizes[i], trainDataset, trainSeqDataset, hyperparameters, trainingOptions);
                    
                    mlpResult[r] = calculationResult["mlp"].trainTimeElapsed;
                    rnnResult[r] = calculationResult["rnn"].trainTimeElapsed;
                    gruResult[r] = calculationResult["gru"].trainTimeElapsed;

                    GC.Collect();
                }

                Array.Sort(mlpResult);
                Array.Sort(rnnResult);
                Array.Sort(gruResult);

                result[hiddenSizes[i]] = new Dictionary<string, double> 
                { 
                    { "mlp", mlpResult[repeats / 2] }, 
                    { "rnn", rnnResult[repeats / 2] }, 
                    { "gru", gruResult[repeats / 2] } 
                };
            }

            return result;
        }

        private static Dictionary<string, FitResponse> Calculation(int windowSize, int hiddenSize,
            Dataset<double[]> trainDataset, Dataset<double[][]> trainSeqDataset,
            Hyperparameters hyperparameters, TrainingOptions trainingOptions)
        {
            MLP mlp = new MLP(windowSize, hiddenSize, 1);
            RNN rnn = new RNN(1, hiddenSize, 1);
            GRU gru = new GRU(1, hiddenSize, 1);

            FitResponse mlpTrainResult = Trainer.Fit(mlp, trainDataset, hyperparameters, trainingOptions);
            FitResponse rnnTrainResult = Trainer.Fit(rnn, trainSeqDataset, hyperparameters, trainingOptions);
            FitResponse gruTrainResult = Trainer.Fit(gru, trainSeqDataset, hyperparameters, trainingOptions);

            return new Dictionary<string, FitResponse> 
            { 
                { "mlp", mlpTrainResult }, 
                { "rnn", rnnTrainResult }, 
                { "gru", gruTrainResult } 
            };
        }

        public static void ShowResultComplexityExperiment(Dictionary<int, Dictionary<string, double>> result)
        {
            Console.WriteLine("Время обучения по размеру скрытого слоя (мс)");
            Console.WriteLine("|" + new string('-', 48) + "|");
            Console.WriteLine($"|{"Скрытый",8} | {"",10} | {"",10} | {"",10} |");
            Console.WriteLine($"|{"слой",8} | {"MLP",10} | {"RNN",10} | {"GRU",10} |");
            Console.WriteLine("|" + new string('-', 48) + "|");

            foreach (int key in result.Keys.OrderBy(k => k))
            {
                var r = result[key];
                Console.WriteLine($"|{key,8} | {r["mlp"],10:F1} | {r["rnn"],10:F1} | {r["gru"],10:F1} |");
            }
            Console.WriteLine("|" + new string('-', 48) + "|");
        }
    }
}
