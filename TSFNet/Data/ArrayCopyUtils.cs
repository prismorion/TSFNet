namespace TSFNet.Data
{
    /// <summary>
    /// Утилиты глубокого копирования зубчатых массивов double (1D, 2D, 3D).
    /// </summary>
    internal static class ArrayCopyUtils
    {
        /// <summary>
        /// Копирование одномерного массива в приёмник.
        /// </summary>
        public static void Copy(double[] source, double[] destination)
        {
            Array.Copy(source, destination, source.Length);
        }

        /// <summary>
        /// Построчное копирование двумерного зубчатого массива в приёмник.
        /// </summary>
        public static void Copy(double[][] source, double[][] destination)
        {
            for (int i = 0; i < source.Length; i++)
                Array.Copy(source[i], destination[i], source[i].Length);
        }

        /// <summary>
        /// Поэлементное копирование трёхмерного зубчатого массива в приёмник.
        /// </summary>
        public static void Copy(double[][][] source, double[][][] destination)
        {
            for (int i = 0; i < source.Length; i++)
                for (int j = 0; j < source[i].Length; j++)
                    Array.Copy(source[i][j], destination[i][j], source[i][j].Length);
        }
    }
}
