namespace TSFNet.Data
{
    /// <summary>
    /// Стандартизация значений (z-нормализация) по среднему и стандартному отклонению,
    /// вычисленным на переданной выборке. Нулевое отклонение заменяется на 1.
    /// </summary>
    public class StandardScaler
    {
        private readonly double mean;
        private readonly double std;

        /// <summary>
        /// Вычисление среднего и стандартного отклонения по переданной выборке.
        /// </summary>
        public StandardScaler(double[] values)
        {
            for (int i = 0; i < values.Length; i++)
                mean += values[i];
            mean /= values.Length;

            for (int i = 0; i < values.Length; i++)
                std += (values[i] - mean) * (values[i] - mean);
            std = Math.Sqrt(std / values.Length);
            std = std < 1e-12 ? 1 : std;
        }

        /// <summary>
        /// Стандартизация одного значения.
        /// </summary>
        public double Transform(double x)
            => (x - mean) / std;

        /// <summary>
        /// Стандартизация массива значений.
        /// </summary>
        public double[] Transform(double[] X)
        {
            double[] res = new double[X.Length];
            for(int i = 0; i < X.Length; i++)
                res[i] = Transform(X[i]);
            return res;
        }

        /// <summary>
        /// Обратное преобразование одного значения в исходный масштаб.
        /// </summary>
        public double InverseTransform(double x)
            => x * std + mean;

        /// <summary>
        /// Обратное преобразование массива значений в исходный масштаб.
        /// </summary>
        public double[] InverseTransform(double[] X)
        {
            double[] res = new double[X.Length];
            for (int i = 0; i < X.Length; i++)
                res[i] = InverseTransform(X[i]);
            return res;
        }
    }
}
