namespace TSFNet.Data
{
    internal static class ArrayCopyUtils
    {
        public static void Copy(double[] source, double[] destination)
        {
            Array.Copy(source, destination, source.Length);
        }

        public static void Copy(double[][] source, double[][] destination)
        {
            for (int i = 0; i < source.Length; i++)
                Array.Copy(source[i], destination[i], source[i].Length);
        }        

        public static void Copy(double[][][] source, double[][][] destination)
        {
            for (int i = 0; i < source.Length; i++)
                for (int j = 0; j < source[i].Length; j++)
                    Array.Copy(source[i][j], destination[i][j], source[i][j].Length);
        }
    }
}
