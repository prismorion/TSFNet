namespace TSFNet.Data
{
    public class StandardScaler
    {
        private readonly double mean;
        private readonly double std;

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

        public double Transform(double x)
            => (x - mean) / std;

        public double[] Transform(double[] X)
        {
            double[] res = new double[X.Length];
            for(int i = 0; i < X.Length; i++)
                res[i] = Transform(X[i]);
            return res;
        }

        public double InverseTransform(double x)
            => x * std + mean;

        public double[] InverseTransform(double[] X)
        {
            double[] res = new double[X.Length];
            for (int i = 0; i < X.Length; i++)
                res[i] = InverseTransform(X[i]);
            return res;
        }
    }
}
