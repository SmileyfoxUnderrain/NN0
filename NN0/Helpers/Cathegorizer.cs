using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Helpers
{
    public class Cathegorizer<T>
    {
        public Cathegorizer(IEnumerable<T> existingCathegories)
        {
            if (existingCathegories == null || existingCathegories.Count() == 0)
                throw new ArgumentException("No cathegories given!");

            ExistingCathegories = existingCathegories.ToList();
        }
        public IList<T> ExistingCathegories { get; private set; }
        public double[] CathegoryToVector(T cathegory)
        {
            if (!ExistingCathegories.Contains(cathegory))
                throw new ArgumentException("Cathegory does not exist");

            var vector = new double[ExistingCathegories.Count()];
            var cathegoryIndex = ExistingCathegories.IndexOf(cathegory);
            vector[cathegoryIndex] = 1;
            return vector;
        }

        public T VectorToCathegory(double[] vector)
        {
            var maxVal = vector.Max();
            var maxValIndex = vector.ToList().IndexOf(maxVal);
            var cathegory = ExistingCathegories.ElementAt(maxValIndex);
            return cathegory;
        }

    }
}
