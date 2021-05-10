using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.LossFunctions
{
    public interface ILossFunction
    {
        /// <summary>
        /// Calculates the loss for the batch of pairs of 
        /// received output value (y) and expected value (d)
        /// </summary>
        /// <param name="receivedToExpectedOutputs">
        /// The enumerable of pairs received to expected output values</param>
        /// <returns>Loss value</returns>
        double CalculateLoss(IEnumerable<Tuple<double, double>> receivedToExpectedOutputs);
    }
}
