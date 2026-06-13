namespace TSFNet.Calculations
{
    /// <summary>
    /// Накопитель суммы квадратов элементов для вычисления L2-нормы градиентов.
    /// </summary>
    public class L2NormAccumulator
    {
        private double sumSq = 0;

        /// <summary>
        /// Добавление квадратов всех элементов переданных векторов в накопитель.
        /// </summary>
        public void AddVectors(params double[][] vector)
        {
            for (int i = 0; i < vector.Length; i++)
                for (int j = 0; j < vector[i].Length; j++)
                    sumSq += vector[i][j] * vector[i][j];
        }

        /// <summary>
        /// Добавление квадратов всех элементов переданных матриц в накопитель.
        /// </summary>
        public void AddMatrices(params double[][][] matrix)
        {
            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < matrix[i].Length; j++)
                    for (int k = 0; k < matrix[i][j].Length; k++)
                        sumSq += matrix[i][j][k] * matrix[i][j][k];
        }

        /// <summary>
        /// Получение L2-нормы — корень из накопленной суммы квадратов.
        /// </summary>
        public double GetNorm() => Math.Sqrt(sumSq);

        /// <summary>
        /// Сброс накопленной суммы квадратов.
        /// </summary>
        public void Reset()
        {
            sumSq = 0;
        }
    }
}
