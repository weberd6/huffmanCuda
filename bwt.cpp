#include <iostream>
#include <cstdlib>

void merge(unsigned char* a[], const int low, const int mid, const int high, int sort_index)
{
    unsigned char **b = new unsigned char*[high+1-low];
    int cur_pos1 = low;
    int cur_pos2 = mid+1;
    int counter = 0;
    int k;
    bool condition;

    // Merges the two array's into b[] until the first one is finish
    while((cur_pos1 <= mid) && (cur_pos2 <= high)) { 
        if (a[cur_pos1][sort_index] <= a[cur_pos2][sort_index]) {
            b[counter] = a[cur_pos1];
            cur_pos1++;
        } else {
            b[counter] = a[cur_pos2];
            cur_pos2++;
        }
        counter++;
    }

    // Completes the array filling in it the missing values
    if(cur_pos1 > mid) {
        for(k = cur_pos2; k <= high; k++) {
            b[counter] = a[k];
            counter++;
        }
    } else {
        for(k = cur_pos1; k <= mid; k++) {
            b[counter] = a[k];
            counter++;
        }
    }

    // Prints into the original array
    for(k = 0; k <= high-low; k++) {
        a[k+low] = b[k];
    }

    delete[] b;
}

void merge_sort(unsigned char* a[], const int low, const int high, int sort_index) {
    int mid;
    if(low < high) {
        mid=(low + high)/2;
        merge_sort(a, low, mid, sort_index);
        merge_sort(a, mid + 1, high, sort_index);
        merge(a, low, mid, high, sort_index);
    }
}

void burrow_wheelers_transform(unsigned char* data_in, unsigned int num_bytes,
                               unsigned char* data_out)
{
    unsigned char** table = new unsigned char*[num_bytes];
    for (int i = 0; i < num_bytes; i++) {
        table[i] = new unsigned char[num_bytes];
    }

    int bytes = num_bytes;
    for (int i = 0; i < num_bytes; i++) {
        for (int j = 0; j < num_bytes; j++) {
            table[i][j] = data_in[((j-i) % bytes + bytes) % bytes];
        }
    }

    for (unsigned int i = 0; i < num_bytes; i++) {
        merge_sort(table, 0, (num_bytes-1), num_bytes-1-i);
    }

    for (int i = 0; i < num_bytes; i++) {
        data_out[i] = table[i][num_bytes-1];
    }

    for (int i = 0; i < num_bytes; i++) {
        delete table[i];
    }
    delete[] table;
}

void inverse_burrow_wheelers_transform(unsigned char* data_in, unsigned int num_bytes,
                                       unsigned char* data_out, unsigned int EOF_char)
{
    unsigned char** table = new unsigned char*[num_bytes];
    for (int i = 0; i < num_bytes; i++) {
        table[i] = new unsigned char[num_bytes];
    }

    for (int i = num_bytes-1; i >= 0; i--) {
        for (int j = 0; j < num_bytes; j++) {
            table[j][i] = data_in[j];
        }

        merge_sort(table, 0, (num_bytes-1), i);
    }

    int return_row;
    for (int i = 0; i < num_bytes; i++) {
        if (table[i][num_bytes-1] == EOF_char) {
            return_row = i;
            break;
        }
    }

    for (int i = 0; i < num_bytes; i++) {
        data_out[i] = table[return_row][i];
    }

    for (int i = 0; i < num_bytes; i++) {
        delete table[i];
    }
    delete[] table;
}

