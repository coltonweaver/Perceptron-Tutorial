package main

import (
    "fmt"
    "math/rand"
    "time"

    "github.com/appliedgo/perceptron/draw"
)

// This is the Perceptron structure. Holds an array of weights and a bias
type Perceptron struct {
    weights []float32
    bias float32
}

// This represents the step function giving a 0 for negatives and 1 for positives
func (p *Perceptron) heaviside(f float32) int32 {
    if f < 0 {
        return 0
    }
    return 1
}

// Create a new perceptron with n inputs. Weights and biases are set to random values
// Between -1 and 1
func NewPerceptron(n int32) *Perceptron {
    var i int32
    w := make([]float32, n, n)
    for i = 0; i < n; i++ {
        w[i] = rand.Float32()*2 - 1
    }
    return &Perceptron{
        weights: w,
        bias: rand.Float32()*2 - 1,
    }
}

// The actual output determination of the perceptron occurs here
// Inputs are taken in, and then sum is made from bias + weights[i]*inputs[i]
// This sum is then ran through the step function
func (p *Perceptron) Process(inputs []int32) int32 {
    sum := p.bias
    for i, input := range inputs {
        sum += float32(input) * p.weights[i]
    }
    return p.heaviside(sum)
}

// Bias and weights are adjusted with respect to difference between prediction and correct answer
// As well as the learning rate
func (p *Perceptron) Adjust(inputs []int32, delta int32, learningRate float32) {
    for i, input := range inputs {
        p.weights[i] += float32(input) * float32(delta) * learningRate
    }
    p.bias += float32(delta) * learningRate
}

/*THIS SECTION IS SETTING UP THE DRAWING OF THE LINE*/

var (
    a, b int32
)

func f(x int32) int32 {
    return a*x+b
}

func isAboveLine(point []int32, f func(int32) int32) int32 {
    x := point[0]
    y := point[1]
    if y > f(x) {
        return 1
    }
    return 0
}

/*THIS SECTION IS FOR TRAINING AND VERIFYING ACCURACY OF THE PREDICTIONS*/

func train(p *Perceptron, iters int, rate float32) {
    // Create random points for each iteration, compare prediction to actual and
    // Adjust model according to difference (delta)
    for i := 0; i < iters; i++ {
        point := []int32{
            rand.Int31n(201) - 101,
            rand.Int31n(201) - 101,
        }

        actual := p.Process(point)
        expected := isAboveLine(point, f)
        delta := expected - actual

        p.Adjust(point, delta, rate)
    }
}

// Take a point at random, verify if it is above or below the line, then predict.
// Keep track of accuracy and plot the point with the correct color
func verify(p *Perceptron) int32 {
    var correctAnswers int32 = 0

    c := draw.NewCanvas()

    for i := 0; i < 100; i++ {

        point := []int32{
            rand.Int31n(201) - 101,
            rand.Int31n(201) - 101,
        }

        result := p.Process(point)

        if result == isAboveLine(point, f) {
            correctAnswers++
        }

        c.DrawPoint(point[0], point[1], result == 1)
    }

    c.DrawLinearFunction(a,b)

    c.Save()

    return correctAnswers
}

func main() {

    rand.Seed(time.Now().UnixNano())
    a = rand.Int31n(11) - 6
    b = rand.Int31n(101) - 51

    p := NewPerceptron(2)

    iterations := 100000
    var learningRate float32 = 0.1

    train(p, iterations, learningRate)

    successRate := verify(p)
    fmt.Printf("%d%% of the answers were correct.\n", successRate)

}
