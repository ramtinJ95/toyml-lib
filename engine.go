package main

import (
	"fmt"
	"math"
	"strings"
)

type Value struct {
	Data     float64
	Grad     float64
	backward func()
	prev     []*Value
	op       string
}

func NewValue(data float64) *Value {
	return &Value{
		Data: data,
		Grad: 0,
	}
}

func (v *Value) Add(other *Value) *Value {
	out := &Value{
		Data: v.Data + other.Data,
		prev: []*Value{v, other},
		op:   "+",
	}
	out.backward = func() {
		v.Grad += 1.0 * out.Grad
		other.Grad += 1.0 * out.Grad
	}
	return out
}

func (v *Value) Neg() *Value {
	return v.Mul(NewValue(-1))
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) Mul(other *Value) *Value {
	out := &Value{
		Data: v.Data * other.Data,
		prev: []*Value{v, other},
		op:   "*",
	}
	out.backward = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

func (v *Value) Pow(other float64) *Value {
	out := &Value{
		Data: math.Pow(v.Data, other),
		prev: []*Value{v},
		op:   "exp",
	}
	out.backward = func() {
		v.Grad += other * (math.Pow(v.Data, (other - 1))) * out.Grad
	}
	return out
}

func (v *Value) Exp() *Value {
	x := v.Data
	out := &Value{
		Data: math.Exp(x),
		prev: []*Value{v},
		op:   "exp",
	}
	out.backward = func() {
		v.Grad += out.Grad * out.Grad
	}
	return out
}

func (v *Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := &Value{
		Data: t,
		prev: []*Value{v},
		op:   "tanh",
	}
	out.backward = func() {
		v.Grad += (1 - t*t) * out.Grad
	}
	return out
}

func (v *Value) Stringify() {
	fmt.Printf("Value:(data = %f), (op = %s)\n", v.Data, v.op)
}

func (v *Value) Graph() string {
	var sb strings.Builder

	var walk func(node *Value, prefix string, isLast bool, isRoot bool)
	walk = func(node *Value, prefix string, isLast bool, isRoot bool) {
		connector := "├── "
		if isLast {
			connector = "└── "
		}
		if isRoot {
			connector = ""
		}

		if node.op != "" {
			sb.WriteString(fmt.Sprintf("%s%s[data: %.4f | grad: %.4f] [%s]\n", prefix, connector, node.Data, node.Grad, node.op))
		} else {
			sb.WriteString(fmt.Sprintf("%s%s[data: %.4f | grad: %.4f]\n", prefix, connector, node.Data, node.Grad))
		}

		var childPrefix string
		if isRoot {
			childPrefix = ""
		} else if isLast {
			childPrefix = prefix + "    "
		} else {
			childPrefix = prefix + "│   "
		}

		for i, child := range node.prev {
			isLastChild := i == len(node.prev)-1
			walk(child, childPrefix, isLastChild, false)
		}
	}

	walk(v, "", true, true)
	return sb.String()
}
