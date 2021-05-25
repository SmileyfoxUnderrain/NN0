using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class BentIdentityFunction : ILayerIndependentFunction
    {
        public double Function(double x)
        {
            var ret = ((Math.Sqrt(Math.Pow(x, 2)) - 1) / 2) + x;
            return ret;
        }
        public double Derivative(double x)
        {
            var ret = (x / 2 * Math.Sqrt(Math.Pow(x, 2) + 1)) + 1;
            return ret;
        }
    }
}
