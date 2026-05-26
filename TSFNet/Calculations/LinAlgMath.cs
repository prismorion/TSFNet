using System.Numerics;
using System.Runtime.InteropServices;

namespace TSFNet.Calculations
{
    /// <summary>
    /// Оптимизированные линейно-алгебраические операции над векторами и матрицами
    /// с использованием SIMD-инструкций (System.Numerics), MemoryMarshal
    /// и множественных аккумуляторов для повышения ILP (Instruction-Level Parallelism).
    /// </summary>
    internal static class LinAlgMath
    {
        /// <summary>
        /// Умножение матрицы на вектор.
        /// Размеры: dest.Length = matrix.Length, vector.Length = matrix[0].Length.
        /// </summary>
        public static void MulMatVec(double[][] matrix, double[] vector, double[] dest)
        {
            int rows = matrix.Length;
            int cols = vector.Length;
            int vecSize = Vector<double>.Count;

            ReadOnlySpan<Vector<double>> vecVecs =
                MemoryMarshal.Cast<double, Vector<double>>(vector);
            int vecCount = vecVecs.Length;
            int scalarStart = vecCount * vecSize;

            for (int i = 0; i < rows; i++)
            {
                double[] mat = matrix[i];
                ReadOnlySpan<Vector<double>> matVecs =
                    MemoryMarshal.Cast<double, Vector<double>>(mat);

                var acc0 = Vector<double>.Zero;
                var acc1 = Vector<double>.Zero;

                int k = 0;
                int limit = vecCount - vecCount % 2;
                for (; k < limit; k += 2)
                {
                    acc0 += matVecs[k] * vecVecs[k];
                    acc1 += matVecs[k + 1] * vecVecs[k + 1];
                }
                if (k < vecCount)
                    acc0 += matVecs[k] * vecVecs[k];

                double sum = Vector.Dot(acc0 + acc1, Vector<double>.One);
                for (int j = scalarStart; j < cols; j++)
                    sum += mat[j] * vector[j];

                dest[i] = sum;
            }
        }

        /// <summary>
        /// Умножение транспонированной матрицы на вектор без явной транспозиции.
        /// Размеры: dest.Length = matrix[0].Length, vector.Length = matrix.Length.
        /// dest зануляется внутри метода (операция аккумуляционная).
        /// </summary>
        public static void MulMatTVec(double[][] matrix, double[] vector, double[] dest)
        {
            int rows = matrix.Length;
            int cols = dest.Length;
            int vecSize = Vector<double>.Count;

            Array.Clear(dest, 0, cols);

            Span<Vector<double>> resVecs =
                MemoryMarshal.Cast<double, Vector<double>>(dest);
            int vecCount = resVecs.Length;
            int scalarStart = vecCount * vecSize;

            for (int j = 0; j < rows; j++)
            {
                double vj = vector[j];
                var scalar = new Vector<double>(vj);
                double[] row = matrix[j];
                ReadOnlySpan<Vector<double>> rowVecs =
                    MemoryMarshal.Cast<double, Vector<double>>(row);

                int k = 0;
                int limit = vecCount - vecCount % 2;
                for (; k < limit; k += 2)
                {
                    resVecs[k] += rowVecs[k] * scalar;
                    resVecs[k + 1] += rowVecs[k + 1] * scalar;
                }
                if (k < vecCount)
                    resVecs[k] += rowVecs[k] * scalar;

                for (int i = scalarStart; i < cols; i++)
                    dest[i] += row[i] * vj;
            }
        }

        /// <summary>
        /// Поэлементное сложение двух векторов.
        /// </summary>
        public static void AddVec(double[] a, double[] b, double[] dest)
        {
            int n = a.Length;
            int vecSize = Vector<double>.Count;

            ReadOnlySpan<Vector<double>> aVecs = MemoryMarshal.Cast<double, Vector<double>>(a);
            ReadOnlySpan<Vector<double>> bVecs = MemoryMarshal.Cast<double, Vector<double>>(b);
            Span<Vector<double>> resVecs = MemoryMarshal.Cast<double, Vector<double>>(dest);

            int vecCount = aVecs.Length;
            int scalarStart = vecCount * vecSize;

            int k = 0;
            int limit = vecCount - vecCount % 2;
            for (; k < limit; k += 2)
            {
                resVecs[k] = aVecs[k] + bVecs[k];
                resVecs[k + 1] = aVecs[k + 1] + bVecs[k + 1];
            }
            if (k < vecCount)
                resVecs[k] = aVecs[k] + bVecs[k];

            for (int i = scalarStart; i < n; i++)
                dest[i] = a[i] + b[i];
        }

        /// <summary>
        /// Поэлементное вычитание двух векторов.
        /// </summary>
        public static void SubVec(double[] a, double[] b, double[] dest)
        {
            int n = a.Length;
            int vecSize = Vector<double>.Count;

            ReadOnlySpan<Vector<double>> aVecs = MemoryMarshal.Cast<double, Vector<double>>(a);
            ReadOnlySpan<Vector<double>> bVecs = MemoryMarshal.Cast<double, Vector<double>>(b);
            Span<Vector<double>> resVecs = MemoryMarshal.Cast<double, Vector<double>>(dest);

            int vecCount = aVecs.Length;
            int scalarStart = vecCount * vecSize;

            int k = 0;
            int limit = vecCount - vecCount % 2;
            for (; k < limit; k += 2)
            {
                resVecs[k] = aVecs[k] - bVecs[k];
                resVecs[k + 1] = aVecs[k + 1] - bVecs[k + 1];
            }
            if (k < vecCount)
                resVecs[k] = aVecs[k] - bVecs[k];

            for (int i = scalarStart; i < n; i++)
                dest[i] = a[i] - b[i];
        }

        /// <summary>
        /// Поэлементное сложение двух матриц.
        /// Все строки dest должны быть предварительно выделены.
        /// </summary>
        public static void AddMat(double[][] a, double[][] b, double[][] dest)
        {
            int rows = a.Length;
            int cols = a[0].Length;
            int vecSize = Vector<double>.Count;
            for (int i = 0; i < rows; i++)
            {
                double[] ai = a[i];
                double[] bi = b[i];
                double[] di = dest[i];
                ReadOnlySpan<Vector<double>> aVecs = MemoryMarshal.Cast<double, Vector<double>>(ai);
                ReadOnlySpan<Vector<double>> bVecs = MemoryMarshal.Cast<double, Vector<double>>(bi);
                Span<Vector<double>> resVecs = MemoryMarshal.Cast<double, Vector<double>>(di);
                int vecCount = aVecs.Length;
                int scalarStart = vecCount * vecSize;
                int k = 0;
                int limit = vecCount - vecCount % 2;
                for (; k < limit; k += 2)
                {
                    resVecs[k] = aVecs[k] + bVecs[k];
                    resVecs[k + 1] = aVecs[k + 1] + bVecs[k + 1];
                }
                if (k < vecCount)
                    resVecs[k] = aVecs[k] + bVecs[k];
                for (int j = scalarStart; j < cols; j++)
                    di[j] = ai[j] + bi[j];
            }
        }

        /// <summary>
        /// Поэлементное вычитание двух матриц.
        /// Все строки dest должны быть предварительно выделены.
        /// </summary>
        public static void SubMat(double[][] a, double[][] b, double[][] dest)
        {
            int rows = a.Length;
            int cols = a[0].Length;
            int vecSize = Vector<double>.Count;

            for (int i = 0; i < rows; i++)
            {
                double[] ai = a[i];
                double[] bi = b[i];
                double[] di = dest[i];

                ReadOnlySpan<Vector<double>> aVecs = MemoryMarshal.Cast<double, Vector<double>>(ai);
                ReadOnlySpan<Vector<double>> bVecs = MemoryMarshal.Cast<double, Vector<double>>(bi);
                Span<Vector<double>> resVecs = MemoryMarshal.Cast<double, Vector<double>>(di);

                int vecCount = aVecs.Length;
                int scalarStart = vecCount * vecSize;

                int k = 0;
                int limit = vecCount - vecCount % 2;
                for (; k < limit; k += 2)
                {
                    resVecs[k] = aVecs[k] - bVecs[k];
                    resVecs[k + 1] = aVecs[k + 1] - bVecs[k + 1];
                }
                if (k < vecCount)
                    resVecs[k] = aVecs[k] - bVecs[k];

                for (int j = scalarStart; j < cols; j++)
                    di[j] = ai[j] - bi[j];
            }
        }

        /// <summary>
        /// Поэлементное умножение двух векторов.
        /// </summary>
        public static void MulElemVec(double[] a, double[] b, double[] dest)
        {
            int n = a.Length;
            int vecSize = Vector<double>.Count;

            ReadOnlySpan<Vector<double>> aVecs = MemoryMarshal.Cast<double, Vector<double>>(a);
            ReadOnlySpan<Vector<double>> bVecs = MemoryMarshal.Cast<double, Vector<double>>(b);
            Span<Vector<double>> resVecs = MemoryMarshal.Cast<double, Vector<double>>(dest);

            int vecCount = aVecs.Length;
            int scalarStart = vecCount * vecSize;

            int k = 0;
            int limit = vecCount - vecCount % 2;
            for (; k < limit; k += 2)
            {
                resVecs[k] = aVecs[k] * bVecs[k];
                resVecs[k + 1] = aVecs[k + 1] * bVecs[k + 1];
            }
            if (k < vecCount)
                resVecs[k] = aVecs[k] * bVecs[k];

            for (int i = scalarStart; i < n; i++)
                dest[i] = a[i] * b[i];
        }

        /// <summary>
        /// Внешнее умножение двух векторов.
        /// Все строки dest должны быть предварительно выделены.
        /// </summary>
        public static void MulOuterVec(double[] a, double[] b, double[][] dest)
        {
            int rows = a.Length;
            int cols = b.Length;
            int vecSize = Vector<double>.Count;

            ReadOnlySpan<Vector<double>> bVecs =
                MemoryMarshal.Cast<double, Vector<double>>(b);
            int vecCount = bVecs.Length;
            int scalarStart = vecCount * vecSize;

            for (int i = 0; i < rows; i++)
            {
                double ai = a[i];
                double[] di = dest[i];
                Span<Vector<double>> resVecs =
                    MemoryMarshal.Cast<double, Vector<double>>(di);
                var aiVec = new Vector<double>(ai);

                int k = 0;
                int limit = vecCount - vecCount % 2;
                for (; k < limit; k += 2)
                {
                    resVecs[k] = aiVec * bVecs[k];
                    resVecs[k + 1] = aiVec * bVecs[k + 1];
                }
                if (k < vecCount)
                    resVecs[k] = aiVec * bVecs[k];

                for (int j = scalarStart; j < cols; j++)
                    di[j] = ai * b[j];
            }
        }

        /// <summary>
        /// Транспонирование матрицы.
        /// Размеры: dest.Length = matrix[0].Length, dest[i].Length = matrix.Length.
        /// Все строки dest должны быть предварительно выделены.
        /// </summary>
        public static void Transposition(double[][] matrix, double[][] dest)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            for (int i = 0; i < cols; i++)
            {
                double[] di = dest[i];
                for (int j = 0; j < rows; j++)
                    di[j] = matrix[j][i];
            }
        }

        /// <summary>
        /// Умножение матрицы на скаляр.
        /// Все строки dest должны быть предварительно выделены.
        /// </summary>
        public static void MulMatNum(double[][] matrix, double number, double[][] dest)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            int vecSize = Vector<double>.Count;
            var num = new Vector<double>(number);

            for (int i = 0; i < rows; i++)
            {
                double[] mi = matrix[i];
                double[] di = dest[i];

                ReadOnlySpan<Vector<double>> mVecs = MemoryMarshal.Cast<double, Vector<double>>(mi);
                Span<Vector<double>> resVecs = MemoryMarshal.Cast<double, Vector<double>>(di);

                int vecCount = mVecs.Length;
                int scalarStart = vecCount * vecSize;

                int k = 0;
                int limit = vecCount - vecCount % 2;
                for (; k < limit; k += 2)
                {
                    resVecs[k] = mVecs[k] * num;
                    resVecs[k + 1] = mVecs[k + 1] * num;
                }
                if (k < vecCount)
                    resVecs[k] = mVecs[k] * num;

                for (int j = scalarStart; j < cols; j++)
                    di[j] = mi[j] * number;
            }
        }

        /// <summary>
        /// Умножение вектора на скаляр.
        /// </summary>
        public static void MulVecNum(double[] a, double number, double[] dest)
        {
            int n = a.Length;
            int vecSize = Vector<double>.Count;
            var num = new Vector<double>(number);

            ReadOnlySpan<Vector<double>> aVecs = MemoryMarshal.Cast<double, Vector<double>>(a);
            Span<Vector<double>> resVecs = MemoryMarshal.Cast<double, Vector<double>>(dest);

            int vecCount = aVecs.Length;
            int scalarStart = vecCount * vecSize;

            int k = 0;
            int limit = vecCount - vecCount % 2;
            for (; k < limit; k += 2)
            {
                resVecs[k] = aVecs[k] * num;
                resVecs[k + 1] = aVecs[k + 1] * num;
            }
            if (k < vecCount)
                resVecs[k] = aVecs[k] * num;

            for (int i = scalarStart; i < n; i++)
                dest[i] = a[i] * number;
        }
    }
}
