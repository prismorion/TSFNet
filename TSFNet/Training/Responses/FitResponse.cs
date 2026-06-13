namespace TSFNet.Training.Responses
{
    /// <summary>
    /// Результат обучения без валидации: затраченное время и журнал ошибки на обучающей выборке по эпохам.
    /// </summary>
    public class FitResponse
    {
        /// <summary> Затраченное на обучение (Fit) время в секундах. </summary>
        public double totalTimeElapsed;

        /// <summary> Затраченное на обучение (model.Train) время в секундах. </summary>
        public double trainTimeElapsed;

        /// <summary> Журнал ошибки на обучающей выборке по эпохам (эпоха → MSE). </summary>
        public Dictionary<int, double> logTrainLoss;

        /// <summary> Журнал ошибки на валидационной выборке по эпохам (эпоха → MSE). </summary>
        public Dictionary<int, double>? logValidationLoss;

        /// <summary>
        /// Заполнение полей результата обучения.
        /// </summary>
        public FitResponse(double totalTimeElapsed, double trainTimeElapsed, 
            Dictionary<int, double> logTrainLoss, Dictionary<int, double>? logValidationLoss = null)
        {
            this.totalTimeElapsed = totalTimeElapsed;
            this.trainTimeElapsed = trainTimeElapsed;
            this.logTrainLoss = logTrainLoss;
            this.logValidationLoss = logValidationLoss;
            this.logValidationLoss = logValidationLoss;
        }
    }
}
