using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Helpers
{
    public class DoublesSeqHelper
    {
        public static void DisplayResult(IEnumerable<double> result)
        {
            Console.WriteLine($"result: {DoublesToString(result)}");
            Console.WriteLine();
        }
        public static string DoublesToString(IEnumerable<double> result)
        {
            var shortResult = result.Select(r => Math.Round(r, 3));
            var ret = string.Join(", ", shortResult);
            return ret;
        }
    }
}
