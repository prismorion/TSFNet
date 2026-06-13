namespace TSFNet.Training.Responses
{
    /// <summary>
    /// Результат обучения с ранней остановкой: время, лучшая эпоха, лучшая ошибка валидации
    /// и журналы ошибок обучения и валидации по эпохам.
    /// </summary>
    public class FitEarlyStoppingResponse
    {
        /// <summary> Затраченное на обучение (FitEarlyStopping) время в секундах. </summary>
        public double totalTimeElapsed;

        /// <summary> Затраченное на обучение (model.Train) время в секундах. </summary>
        public double trainTimeElapsed;

        /// <summary> Эпоха с наименьшей ошибкой на валидации. </summary>
        public int bestEpoch;

        /// <summary> Наименьшая достигнутая ошибка на валидации. </summary>
        public double bestValidationLoss;

        /// <summary> Журнал ошибки на обучающей выборке по эпохам (эпоха → MSE). </summary>
        public Dictionary<int, double> logTrainLoss;

        /// <summary> Журнал ошибки на валидации по эпохам (эпоха → MSE). </summary>
        public Dictionary<int, double> logValidationLoss;

        /// <summary>
        /// Заполнение полей результата обучения с ранней остановкой.
        /// </summary>
        public FitEarlyStoppingResponse(double totalTimeElapsed, double trainTimeElapsed, int bestEpoch, double bestValidationLoss,
            Dictionary<int, double> logTrainLoss, Dictionary<int, double> logValidationLoss)
        {
            this.totalTimeElapsed = totalTimeElapsed;
            this.trainTimeElapsed = trainTimeElapsed;
            this.bestEpoch = bestEpoch;
            this.bestValidationLoss = bestValidationLoss;
            this.logTrainLoss = logTrainLoss;
            this.logValidationLoss = logValidationLoss;
        }
    }
}
