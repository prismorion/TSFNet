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
        public TrainingOptions(int _epochs = 100, int _reportEvery = 0, int _patience = 20)
        {
            epochs = _epochs;
            reportEvery = _reportEvery;
            patience = _patience;
        }
    }
}
