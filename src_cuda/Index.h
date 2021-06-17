#ifndef INDEX_H_
#define INDEX_H_

#include "defines.h"

namespace index {
    inline int index_xy(int i, int j) {
        return  ( i + defines::nx* j );
    };
};
#endif


