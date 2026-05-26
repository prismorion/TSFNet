namespace TSFNet.Data
{
    public static class Differencer
    {
        public static double[] Diff(double[] values)
        {
            double[] res = new double[values.Length - 1];
            for (int i = 0; i < values.Length - 1; i++)
                res[i] = values[i + 1] - values[i];
            return res;
        }

        public static double[] InverseDiff(double[] values, double anchor)
        {
            double[] res = new double[values.Length + 1];
            res[0] = anchor;
            for (int i = 0; i < values.Length; i++)
                res[i + 1] = res[i] + values[i];
            return res;
        }
    }
}
