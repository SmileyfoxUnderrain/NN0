using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.ActivationFunctions
{
    class TahnFunction : ILayerIndependentFunction
    {
        public double Function(double x)
        {
            return Math.Tanh(x);
        }
        public double Derivative(double x)
        {
            var ret = 1 - Math.Pow(x, 2);
            return ret;
        }
    }
}
