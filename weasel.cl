__kernel void mutate(__global char* sentence, __global char* target_sentence, int length, __global int* scores, __global char* mutated_sentences) {
    int i = get_global_id(0);
    int score = 0;
    int offset = i * length;
    char alphabets[27] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    #pragma unroll
    for(int j=0; j<length; j++) {
        uint seed = i;
        uint t = seed ^ (seed << 11);
        uint result = 3 ^ (3 >> 19) ^ (3 ^ (3 >> 8))
     //   printf("Random %c\n", alphabets[random_number]);
        char letter = alphabets[result % 27];
        if(target_sentence[j] == letter) score++;
        mutated_sentences[j+offset] = letter;
    }
    scores[i] = score;
}
