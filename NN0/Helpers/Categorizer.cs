using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Helpers
{
    public class Categorizer<T>
    {
        public Categorizer(IEnumerable<T> existingCategories)
        {
            if (existingCategories == null || existingCategories.Count() == 0)
                throw new ArgumentException("No categories given!");

            ExistingCategories = existingCategories.ToList();
        }
        public IList<T> ExistingCategories { get; private set; }
        public IEnumerable<double> CategoryToVector(T category)
        {
            if (!ExistingCategories.Contains(category))
                throw new ArgumentException("Category does not exist");

            var vector = new double[ExistingCategories.Count()];
            var categoryIndex = ExistingCategories.IndexOf(category);
            vector[categoryIndex] = 1;
            return vector;
        }

        public T VectorToCategory(IEnumerable<double> vector)
        {
            var maxVal = vector.Max();
            var maxValIndex = vector.ToList().IndexOf(maxVal);
            var category = ExistingCategories.ElementAt(maxValIndex);
            return category;
        }

    }
}
