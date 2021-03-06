#include <iostream>

void move_to_front_transform(unsigned char* data_in, unsigned int num_bytes,
                             unsigned char* sequence)
{
    const unsigned int NUM_VALS = 256;
    unsigned char list[NUM_VALS];
    for (unsigned int i = 0; i < NUM_VALS; i++) {
        list[i] = i;
    }

    for (unsigned int i = 0; i < num_bytes; i++) {
        unsigned char current_char = data_in[i];

        for (unsigned int j = 0; j < NUM_VALS; j++) {
            if (current_char == list[j]) {
                sequence[i] = j;

                for (unsigned int k = j; k > 0; k--) {
		    list[k] = list[k-1];
                }

                list[0] = data_in[i];
                break;
            }
        }
    }
}

void inverse_move_to_front_transform(unsigned char* sequence, unsigned int num_bytes,
                                     unsigned char* data_out)
{
    const unsigned int NUM_VALS = 256;
    unsigned int list[NUM_VALS];
    for (unsigned int i = 0; i < NUM_VALS; i++) {
        list[i] = i;
    }

    for (unsigned int i = 0; i < num_bytes; i++) {
        data_out[i] = list[sequence[i]];

        for (unsigned int j = sequence[i]; j > 0; j--) {
            list[j] = list[j-1];
        }

        list[0] = data_out[i];
    }
}

