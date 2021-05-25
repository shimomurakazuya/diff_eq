#include "Index.h"
#include "defines.h"

namespace index{
    int index_xy(int  i , int j){
        return  ( j + defines::ny* i );
}
};



