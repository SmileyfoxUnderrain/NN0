using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    class TahnFunction : IActivationFunction
    {
        public double Function(double x)
        {
            return Math.Tanh(x);
            //var e = Math.E;
            //var minusX = x * -1;
            //var ret = (Math.Pow(e, x) - Math.Pow(e, minusX)) / (Math.Pow(e, x) + Math.Pow(e, minusX));
            //return ret;
        }
        public double Derivative(double x)
        {
            var ret = 1 - Math.Pow(x, 2);
            return ret;
        }
    }
}
