using TSFNet.Data;

namespace TSFNet.Calculations
{
    /// <summary>
    /// Метрики качества регрессии (MSE, RMSE, MAE) для отдельных примеров и для целого датасета.
    /// </summary>
    public static class Metrics
    {
        /// <summary>
        /// Среднеквадратичная ошибка (MSE) между истинным и предсказанным вектором.
        /// </summary>
        public static double MSE(double[] yTrue, double[] yPred)
        {
            int n = yTrue.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double d = yPred[i] - yTrue[i];
                sum += d * d;
            }
            return sum / n;
        }

        /// <summary>
        /// Средний MSE по всему датасету для заданной функции предсказания.
        /// </summary>
        /// <param name="predict">Функция предсказания модели.</param>
        public static double MSE<TInput>(Dataset<TInput> dataset, Func<TInput, double[]> predict)
        {
            double sum = 0;
            for (int i = 0; i < dataset.Length; i++)
                sum += MSE(dataset.GetTarget(i), predict(dataset.GetInput(i)));
            return sum / dataset.Length;
        }

        /// <summary>
        /// Корень из среднеквадратичной ошибки (RMSE) для пары векторов.
        /// </summary>
        public static double RMSE(double[] yTrue, double[] yPred)
            => Math.Sqrt(MSE(yTrue, yPred));

        /// <summary>
        /// Корень из среднего MSE (RMSE) по всему датасету для заданной функции предсказания.
        /// </summary>
        /// <param name="predict">Функция предсказания модели.</param>
        public static double RMSE<TInput>(Dataset<TInput> dataset, Func<TInput, double[]> predict)
            => Math.Sqrt(MSE(dataset, predict));

        /// <summary>
        /// Средняя абсолютная ошибка (MAE) между истинным и предсказанным вектором.
        /// </summary>
        public static double MAE(double[] yTrue, double[] yPred)
        {
            int n = yTrue.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += Math.Abs(yPred[i] - yTrue[i]);
            return sum / n;
        }

        /// <summary>
        /// Средняя абсолютная ошибка (MAE) по всему датасету для заданной функции предсказания.
        /// </summary>
        /// <param name="predict">Функция предсказания модели.</param>
        public static double MAE<TInput>(Dataset<TInput> dataset, Func<TInput, double[]> predict)
        {
            double sum = 0;
            for (int i = 0; i < dataset.Length; i++)
                sum += MAE(dataset.GetTarget(i), predict(dataset.GetInput(i)));
            return sum / dataset.Length;
        }
    }
}
