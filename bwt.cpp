void merge(unsigned char* a[], const int low, const int mid, const int high, int sort_index)
{
    unsigned char **b = new unsigned char*[high + 1 - low];
    int k;
    int h = low;
    int i = 0;
    int j = mid + 1;

    // Merges the two array's into b[] until the first one is finish
    while((h <= mid) && (j <= high)) {
        if(a[h][sort_index] <= a[j][sort_index]) {
            b[i] = a[h];
            h++;
        } else {
            b[i] = a[j];
            j++;
        }
        i++;
    }

    // Completes the array filling in it the missing values
    if(h > mid) {
        for(k = j; k <= high; k++) {
            b[i] = a[k];
            i++;
        }
    } else {
        for(k = h; k <= mid; k++) {
            b[i] = a[k];
            i++;
        }
    }

    // Prints into the original array
    for(k = 0; k <= high - low; k++) {
        a[k + low] = b[k];
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
    for (unsigned int i = 0; i < num_bytes; i++) {
        table[i] = new unsigned char[num_bytes];
    }

    for (unsigned int i = 0; i < num_bytes; i++) {
        for (unsigned int j = 0; j < num_bytes; j++) {
            table[i][j] = data_in[(i+j) % num_bytes];
        }
    }

    // Swap last row with 2nd row to put EOF char at beginning of last row
    unsigned char* temp = table[num_bytes-1];
    table[num_bytes] = table[1];
    table[1] = temp;

    // Don't include last row
    merge_sort(table, 0, num_bytes-2, 0);

    for (unsigned int i = 0; i <= num_bytes; i++) {
        data_out[i] = table[i][num_bytes];
    }
}

void inverse_burrow_wheelers_transform(unsigned char* data_in, unsigned int num_bytes,
                                       unsigned char* data_out, unsigned char EOF_char)
{
    unsigned char** table = new unsigned char*[num_bytes];
    for (unsigned int i = 0; i < num_bytes; i++) {
        table[i] = new unsigned char[num_bytes];
    }

    unsigned int magic_index;
    for (unsigned int i = 0; i < num_bytes; i++) {
        if (EOF_char == data_in[i]) {
            magic_index = i;
            break;
        }
    }

    for (unsigned int i = num_bytes-1; i >= 0; i--) {
        for (unsigned int j = 0; j < num_bytes; j++) {
            table[i][j] = data_in[j];
        }

        unsigned char* temp = table[num_bytes-1];
        table[num_bytes] = table[magic_index];
        table[magic_index] = temp;

        merge_sort(table, 0, num_bytes-2, i);
    }

    for (unsigned int i = 0; i < num_bytes; i++) {
        data_out[i] = table[magic_index][i];
    }
}

