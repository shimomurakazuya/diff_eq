#include "Index.h"
#include "defines.h"

namespace Index{
    int index_xy(int  i , int j){
        return  ( (j+1) + defines::ny* (i+1) );
}
};



