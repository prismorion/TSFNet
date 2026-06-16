namespace TSFNet.Calculations
{
    public static class RandGen
    {
        private static Random? seeded;

        /// <summary>
        /// Текущий генератор. Если seed не задан — обычный потокобезопасный Random.Shared,
        /// иначе — детерминированный экземпляр с зафиксированным seed.
        /// </summary>
        public static Random Shared => seeded ?? Random.Shared;

        /// <summary>
        /// Фиксирует seed для воспроизводимости результатов.
        /// </summary>
        public static void Seed(int seed) => seeded = new Random(seed);

        /// <summary>
        /// Возврат к обычному (несидированному) рандому.
        /// </summary>
        public static void Reset() => seeded = null;
    }
}