using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.LossFunctions
{
    public class SquaredHingeFunction : ILossFunction
    {
        public double CalculateLoss(IEnumerable<Tuple<double, double>> receivedToExpectedOutputs)
        {
            var sum = receivedToExpectedOutputs
                .Sum(t => Math.Pow(Math.Max(1 - t.Item1 * t.Item2, 0), 2));
            var ret = sum / receivedToExpectedOutputs.Count();
            return ret;
        }
    }
}
