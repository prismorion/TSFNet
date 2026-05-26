using TSFNet.Data;

namespace TSFNet.Calculations
{
    public static class Metrics
    {
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

        public static double MSE<TInput>(Dataset<TInput> dataset, Func<TInput, double[]> predict)
        {
            double sum = 0;
            for (int i = 0; i < dataset.Length; i++)
                sum += MSE(dataset.GetTarget(i), predict(dataset.GetInput(i)));
            return sum / dataset.Length;
        }

        public static double RMSE(double[] yTrue, double[] yPred)
            => Math.Sqrt(MSE(yTrue, yPred));

        public static double RMSE<TInput>(Dataset<TInput> dataset, Func<TInput, double[]> predict)
            => Math.Sqrt(MSE(dataset, predict));

        public static double MAE(double[] yTrue, double[] yPred)
        {
            int n = yTrue.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += Math.Abs(yPred[i] - yTrue[i]);
            return sum / n;
        }

        public static double MAE<TInput>(Dataset<TInput> dataset, Func<TInput, double[]> predict)
        {
            double sum = 0;
            for (int i = 0; i < dataset.Length; i++)
                sum += MAE(dataset.GetTarget(i), predict(dataset.GetInput(i)));
            return sum / dataset.Length;
        }
    }
}
