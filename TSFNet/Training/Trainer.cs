using System.Diagnostics;
using TSFNet.Calculations;
using TSFNet.Data;
using TSFNet.Training.Parameters;
using TSFNet.Training.Responses;

namespace TSFNet.Training
{
    public static class Trainer
    {
        public static FitResponse Fit<TInput, TBuffer, TSnapshot>(ITrainable<TInput, TBuffer, TSnapshot> model, Dataset<TInput> trainDataset,
            Hyperparameters hyperparameters, TrainingOptions trainingOptions)
        {
            Stopwatch sw = Stopwatch.StartNew();

            Dictionary<int, double> logTrainLoss = new Dictionary<int, double>();

            if (trainingOptions.reportEvery > 0)
                logTrainLoss[0] = Metrics.MSE(trainDataset, model.Forward);
            
            TBuffer buffer = model.CreateBuffer(trainDataset, hyperparameters);

            // проход по эпохам
            for (int e = 1; e <= trainingOptions.epochs; e++)
            {
                trainDataset.Shuffle();
                model.Train(trainDataset, hyperparameters, buffer);

                if (trainingOptions.reportEvery > 0 && e % trainingOptions.reportEvery == 0)
                    logTrainLoss[e] = Metrics.MSE(trainDataset, model.Forward);
            }
            trainDataset.ResetOrder();

            sw.Stop();
            double timeElapsed = sw.Elapsed.TotalSeconds;

            return new FitResponse(timeElapsed, logTrainLoss);
        }

        public static FitEarlyStoppingResponse Fit<TInput, TBuffer, TSnapshot>(ITrainable<TInput, TBuffer, TSnapshot> model, 
            Dataset<TInput> trainDataset, Dataset<TInput> validationDataset, 
            Hyperparameters hyperparameters, TrainingOptions trainingOptions)
        {
            Stopwatch sw = Stopwatch.StartNew();

            Dictionary<int, double> logTrainLoss = new Dictionary<int, double>();
            Dictionary<int, double> logValidationLoss = new Dictionary<int, double>();

            TBuffer buffer = model.CreateBuffer(trainDataset, hyperparameters);
            TSnapshot snapshot = model.CreateSnapshotBuffer();

            double trainLoss = Metrics.MSE(trainDataset, model.Forward);
            double validationLoss = Metrics.MSE(validationDataset, model.Forward);
            double bestValidationLoss = validationLoss;
            int bestEpoch = 0;
            int epochsWithoutImprovement = 0;

            model.SaveSnapshot(snapshot);

            if (trainingOptions.reportEvery > 0)
            {
                logTrainLoss[0] = trainLoss;
                logValidationLoss[0] = validationLoss;
            }
            
            // проход по эпохам
            for (int e = 1; e <= trainingOptions.epochs; e++)
            {
                trainDataset.Shuffle();
                model.Train(trainDataset, hyperparameters, buffer);
                trainLoss = Metrics.MSE(trainDataset, model.Forward);
                validationLoss = Metrics.MSE(validationDataset, model.Forward);

                if(validationLoss < bestValidationLoss)
                {
                    bestValidationLoss = validationLoss;
                    model.SaveSnapshot(snapshot);
                    bestEpoch = e;
                    epochsWithoutImprovement = 0;
                }
                else
                    epochsWithoutImprovement++;

                if (trainingOptions.reportEvery > 0 && e % trainingOptions.reportEvery == 0)
                {
                    logTrainLoss[e] = trainLoss;
                    logValidationLoss[e] = validationLoss;
                }

                if (epochsWithoutImprovement >= trainingOptions.patience)
                {
                    if (trainingOptions.reportEvery > 0 && !logValidationLoss.ContainsKey(e))
                    {
                        logTrainLoss[e] = trainLoss;
                        logValidationLoss[e] = validationLoss;
                    }
                    break;
                }
            }
            model.RestoreSnapshot(snapshot);
            trainDataset.ResetOrder();            

            sw.Stop();
            double timeElapsed = sw.Elapsed.TotalSeconds;

            return new FitEarlyStoppingResponse(timeElapsed, bestEpoch, bestValidationLoss, logTrainLoss, logValidationLoss);
        }
    }
}
