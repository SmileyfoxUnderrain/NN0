using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    public class GaussianFunction : ILayerIndependentFunction
    {
        public double Function(double x)
        {
            var ret = Math.Pow(Math.E, Math.Pow(-1 * x, 2));
            return ret;
        }
        public double Derivative(double x)
        {
            var ret = -2 * x * Math.Pow(Math.E, Math.Pow(-1 * x, 2));
            return ret;
        }
    }
}
