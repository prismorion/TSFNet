using System.Diagnostics;
using TSFNet.Calculations;
using TSFNet.Data;
using TSFNet.Training.Parameters;
using TSFNet.Training.Responses;

namespace TSFNet.Training
{
    /// <summary>
    /// Универсальный цикл обучения моделей: проход по эпохам, перемешивание выборки,
    /// логирование ошибки и ранняя остановка.
    /// </summary>
    public static class Trainer
    {
        /// <summary>
        /// Обучение модели заданное число эпох с перемешиванием выборки и опциональным логированием ошибки.
        /// Возвращает затраченное время и журнал ошибки на обучающей выборке.
        /// </summary>
        public static FitResponse Fit<TInput, TBuffer, TSnapshot>(ITrainable<TInput, TBuffer, TSnapshot> model, 
            Dataset<TInput> trainDataset, Hyperparameters hyperparameters, TrainingOptions trainingOptions)
        {
            Stopwatch swTotal = Stopwatch.StartNew();

            Dictionary<int, double> logTrainLoss = new Dictionary<int, double>();

            if (trainingOptions.reportEvery > 0)
                logTrainLoss[0] = Metrics.MSE(trainDataset, model.Predict);
            
            TBuffer buffer = model.CreateBuffer(trainDataset, hyperparameters);

            // проход по эпохам
            Stopwatch swTrain = new Stopwatch();
            for (int e = 1; e <= trainingOptions.epochs; e++)
            {
                trainDataset.Shuffle();

                swTrain.Start();
                model.Train(trainDataset, hyperparameters, buffer);
                swTrain.Stop();

                if (trainingOptions.reportEvery > 0 && e % trainingOptions.reportEvery == 0)
                    logTrainLoss[e] = Metrics.MSE(trainDataset, model.Predict);
            }
            trainDataset.ResetOrder();

            swTotal.Stop();

            double totalTimeElapsed = swTotal.Elapsed.TotalMilliseconds;
            double trainTimeElapsed = swTrain.Elapsed.TotalMilliseconds;

            return new FitResponse(totalTimeElapsed, trainTimeElapsed, logTrainLoss);
        }

        /// <summary>
        /// Обучение модели заданное число эпох с перемешиванием выборки и опциональным логированием ошибки.
        /// Возвращает затраченное время и журналы ошибки на обучающей выборке.
        /// </summary>
        public static FitResponse Fit<TInput, TBuffer, TSnapshot>(ITrainable<TInput, TBuffer, TSnapshot> model, 
            Dataset<TInput> trainDataset, Dataset<TInput> validationDataset,
            Hyperparameters hyperparameters, TrainingOptions trainingOptions)
        {
            Stopwatch swTotal = Stopwatch.StartNew();

            Dictionary<int, double> logTrainLoss = new Dictionary<int, double>();
            Dictionary<int, double> logValidationLoss = new Dictionary<int, double>();

            if (trainingOptions.reportEvery > 0)
            {
                logTrainLoss[0] = Metrics.MSE(trainDataset, model.Predict);
                logValidationLoss[0] = Metrics.MSE(validationDataset, model.Predict);
            }

            TBuffer buffer = model.CreateBuffer(trainDataset, hyperparameters);

            // проход по эпохам
            Stopwatch swTrain = new Stopwatch();
            for (int e = 1; e <= trainingOptions.epochs; e++)
            {
                trainDataset.Shuffle();

                swTrain.Start();
                model.Train(trainDataset, hyperparameters, buffer);
                swTrain.Stop();

                if (trainingOptions.reportEvery > 0 && e % trainingOptions.reportEvery == 0)
                {
                    logTrainLoss[e] = Metrics.MSE(trainDataset, model.Predict);
                    logValidationLoss[e] = Metrics.MSE(validationDataset, model.Predict);
                }
            }
            trainDataset.ResetOrder();

            swTotal.Stop();

            double totalTimeElapsed = swTotal.Elapsed.TotalMilliseconds;
            double trainTimeElapsed = swTrain.Elapsed.TotalMilliseconds;

            return new FitResponse(totalTimeElapsed, trainTimeElapsed, logTrainLoss, logValidationLoss);
        }

        /// <summary>
        /// Обучение с ранней остановкой: контроль ошибки на валидации, сохранение снимка лучшего состояния
        /// и его восстановление по завершении. Возвращает время, лучшую эпоху, лучшую ошибку и журналы ошибок.
        /// </summary>
        public static FitEarlyStoppingResponse FitEarlyStopping<TInput, TBuffer, TSnapshot>(ITrainable<TInput, TBuffer, TSnapshot> model, 
            Dataset<TInput> trainDataset, Dataset<TInput> validationDataset, 
            Hyperparameters hyperparameters, TrainingOptions trainingOptions)
        {
            Stopwatch swTotal = Stopwatch.StartNew();

            Dictionary<int, double> logTrainLoss = new Dictionary<int, double>();
            Dictionary<int, double> logValidationLoss = new Dictionary<int, double>();

            TBuffer buffer = model.CreateBuffer(trainDataset, hyperparameters);
            TSnapshot snapshot = model.CreateSnapshotBuffer();

            double trainLoss = Metrics.MSE(trainDataset, model.Predict);
            double validationLoss = Metrics.MSE(validationDataset, model.Predict);
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
            Stopwatch swTrain = new Stopwatch();
            for (int e = 1; e <= trainingOptions.epochs; e++)
            {
                trainDataset.Shuffle();

                swTrain.Start();
                model.Train(trainDataset, hyperparameters, buffer);
                swTrain.Stop();

                trainLoss = Metrics.MSE(trainDataset, model.Predict);
                validationLoss = Metrics.MSE(validationDataset, model.Predict);

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

            swTotal.Stop();
            double totalTimeElapsed = swTotal.Elapsed.TotalMilliseconds;
            double trainTimeElapsed = swTrain.Elapsed.TotalMilliseconds;

            return new FitEarlyStoppingResponse(totalTimeElapsed, trainTimeElapsed, bestEpoch, bestValidationLoss, logTrainLoss, logValidationLoss);
        }
    }
}
