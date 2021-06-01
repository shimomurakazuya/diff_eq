#include "Index.h"
#include "defines.h"

namespace index{
    int index_xy(int  i , int j){
        return  ( i + defines::nx* j );
}
};



