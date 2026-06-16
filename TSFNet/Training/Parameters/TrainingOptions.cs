namespace TSFNet.Training.Parameters
{
    /// <summary>
    /// Параметры процесса обучения: число эпох, частота логирования и терпение ранней остановки.
    /// </summary>
    public class TrainingOptions
    {
        /// <summary> Количество эпох обучения. </summary>
        public int epochs { get; set; }

        // <summary> Частота логирования ошибки в эпохах (0 - логирование выключено). </summary>
        public int reportEvery { get; set; }

        /// <summary> Число эпох без улучшения на валидации до ранней остановки. </summary>
        public int patience { get; set; }

        /// <summary>
        /// Создание набора опций обучения со значениями по умолчанию.
        /// </summary>
        public TrainingOptions(int epochs = 100, int reportEvery = 0, int patience = 20)
        {
            this.epochs = epochs;
            this.reportEvery = reportEvery;
            this.patience = patience;
        }
    }
}
