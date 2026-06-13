namespace TSFNet.Data
{
    /// <summary>
    /// Дифференцирование и обратное интегрирование временного ряда
    /// (приведение к стационарности и восстановление исходных значений).
    /// </summary>
    public static class Differencer
    {
        /// <summary>
        /// Первая разность ряда: разности соседних элементов. Длина результата на 1 меньше.
        /// </summary>
        public static double[] Diff(double[] values)
        {
            double[] res = new double[values.Length - 1];
            for (int i = 0; i < values.Length - 1; i++)
                res[i] = values[i + 1] - values[i];
            return res;
        }

        /// <summary>
        /// Восстановление исходного ряда из разностей по якорному начальному значению.
        /// </summary>
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
