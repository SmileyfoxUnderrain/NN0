using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN0.Functions
{
    public interface IActivationFunction
    {
        double Function(double x);
        double Derivative(double x);
    }
}
