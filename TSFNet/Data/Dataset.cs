namespace TSFNet.Data
{
    /// <summary>
    /// Контейнер обучающей выборки: пары "вход - целевое значение" с обходом
    /// через индексный массив порядка (поддержка перемешивания без копирования данных).
    /// </summary>
    public class Dataset<TInput>
    {
        private TInput[] inputs;
        private double[][] targets;
        private int[] order;
        private static Random rand = new Random();

        /// <summary>
        /// Создание датасета из массивов входов и целевых значений; инициализация порядка обхода.
        /// </summary>
        public Dataset(TInput[] inputs, double[][] targets)
        {
            this.inputs = inputs;
            this.targets = targets;
            order = new int[inputs.Length];
            ResetOrder();
        }

        /// <summary>
        /// Количество примеров в датасете.
        /// </summary>
        public int Length => inputs.Length;

        /// <summary>
        /// Вход i-го примера согласно текущему порядку обхода.
        /// </summary>
        public TInput GetInput(int i) => inputs[order[i]];

        /// <summary>
        /// Целевое значение i-го примера согласно текущему порядку обхода.
        /// </summary>
        public double[] GetTarget(int i) => targets[order[i]];

        /// <summary>
        /// Исходный массив входов без учёта порядка обхода.
        /// </summary>
        public TInput[] GetRawInput => inputs;

        /// <summary>
        /// Исходный массив целевых значений без учёта порядка обхода.
        /// </summary>
        public double[][] GetRawTarget => targets;        

        /// <summary>
        /// Случайное перемешивание порядка обхода (алгоритм Фишера–Йейтса).
        /// </summary>
        public void Shuffle()
        {
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rand.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }

        /// <summary>
        /// Сброс порядка обхода к исходному.
        /// </summary>
        public void ResetOrder()
        {
            for (int i = 0; i < inputs.Length; i++)
                order[i] = i;
        }
    }
}
