#include "darknet.h"
#include "fix8.h"
#include "fl8.h"

QUANTIZE_TYPE get_type(char *type_str)
{
  if ( 0 == strcmp(type_str, "FL32") || 0 == strcmp(type_str, "fl32") )
    return FL32;
  else if ( 0 == strcmp(type_str, "FIX8") || 0 == strcmp(type_str, "fix8") )
    return FIX8;
  else if ( 0 == strcmp(type_str, "FL8_M0E7") || 0 == strcmp(type_str, "fl8_m0e7") )
    return FL8_M0E7;
  else if ( 0 == strcmp(type_str, "FL8_M1E6") || 0 == strcmp(type_str, "fl8_m1e6") )
    return FL8_M1E6;
  else if ( 0 == strcmp(type_str, "FL8_M2E5") || 0 == strcmp(type_str, "fl8_m2e5") )
    return FL8_M2E5;
  else if ( 0 == strcmp(type_str, "FL8_M3E4") || 0 == strcmp(type_str, "fl8_m3e4") )
    return FL8_M3E4;
  else if ( 0 == strcmp(type_str, "FL8_M4E3") || 0 == strcmp(type_str, "fl8_m4e3") )
    return FL8_M4E3;
  else if ( 0 == strcmp(type_str, "FL8_M5E2") || 0 == strcmp(type_str, "fl8_m5e2") )
    return FL8_M5E2;
  else if ( 0 == strcmp(type_str, "FL8_M6E1") || 0 == strcmp(type_str, "fl8_m6e1") )
    return FL8_M6E1;
  else if ( 0 == strcmp(type_str, "FL8_M7E0") || 0 == strcmp(type_str, "fl8_m7e0") )
    return FL8_M7E0;
  else
    return FL32;
}

void run_quantize(int argc, char **argv)
{
    if(argc < 5){
        fprintf(stderr, "usage: %s %s [fix8/fl8_m3e4/fl8_m4e3] [cfg] [weights] [img]\n", argv[0], argv[1]);
        return;
    }

    char *type_str = argv[2];
    char *cfg = argv[3];
    char *weights = argv[4];
    char *img = argv[5];

    QUANTIZE_TYPE type = get_type(type_str);
    
    char *q_method;
    if (type == FL32)
      q_method = "fl32";
    else if (type == FIX8)
      q_method = "fix8";
    else if (type >= 2 && type <= 9)
      q_method = "fl8";
    
    if ( 0 == strcmp(q_method, "fix8") ) {
      printf("Quantize to %s ...\n", type_str);
      //fix8_quantize(cfg, weights, img, "fix8.weights", "fix8.decision");
    }
    else if ( 0 == strcmp(q_method, "fl8") ) {
      printf("Quantize to %s ...\n", type_str);
      //fl8_quantize(cfg, weights, img, type);
    }
}

