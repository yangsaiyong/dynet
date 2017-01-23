/**
 * \file expr-seq.h
 * \brief A wrapper around a sequence of expressions.
 * 
 */

#ifndef DYNET_EXPR_SEQ_H
#define DYNET_EXPR_SEQ_H

#include "dynet/expr.h"

namespace dynet {
  namespace expr {

class ExpressionSequence {

  /**
  * Initialize an expression sequence with a vector of expressions. They should
  * all be of the same dimensions.
  */
  ExpressionSequence(const std::vector<Expression> & t) : tens(), vec(v) { }

  /**
  * Initialize an expression sequence a tensor where the highest dimension
  * represents "time steps".
  */
  ExpressionSequence(const Expression & t) : tens(t), vec(t.d[t.d.nd-1]) { }
 
  /**
  * Return the number of steps in the sequence.
  */
  size_t size() { return vec.size(); }

  /**
  * Get the expression at the idxth position in the sequence.
  */
  const Expression& get(size_t idx) {
    if(idx > vec.size())
      throw std::invalid_argument("Out of bounds in ExpressionSequence::get()");
    if(vec[idx].pg == nullptr)
      vec[idx] = pick(tens, idx, tens.d.nd-1);
    return vec[idx];
  }

  /**
  * Get the expression at the idxth position in the sequence.
  */
  const Expression& operator[](size_t idx) { return get(idx); }

  /**
  * Get the expression as a tensor.
  */
  const Expression& as_tensor() {
    if(tens.pg == nullptr)
      tens = join(vec);
    return tens;
  }


protected:
  Expression tens;
  std::vector<Expression> vec;

}

  }
}

#endif
