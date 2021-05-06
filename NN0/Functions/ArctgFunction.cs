using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    public class ArctgFunction : IActivationFunction
    {
        public double Function(double x)
        {
            var ret = Math.Pow(Math.Tan(x), -1);
            return ret;
        }
        public double Derivative(double x)
        {
            var ret = 1 / (Math.Pow(x, 2) + 1);
            return ret;
        }
    }
}
