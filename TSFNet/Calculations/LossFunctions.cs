namespace TSFNet.Calculations
{
    public static class LossFunctions
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
            return sum / (2 * n);
        }

        public static void MSEDerivative(double[] yTrue, double[] yPred, double[] dest)
        {
            int n = yTrue.Length;
            for (int i = 0; i < n; i++)
                dest[i] = (yPred[i] - yTrue[i]) / n;
        }
    }
}
