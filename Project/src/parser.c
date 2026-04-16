#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int duration;
  int *resources;
  int num_successors;
  int *successors;
} Activity;

typedef struct {
  int n;
  int K;
  Activity *activities;
  int *resource_capacities;
} Project;

Project parse_psplib(const char *filename);

Project parse_psplib(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Invalid file name/path!");
    exit(EXIT_FAILURE);
  }
  Project project;
  fscanf(file, "%d %d", &project.n, &project.K);
  project.activities = malloc(sizeof(Activity) * (project.n + 2));
  project.resource_capacities = malloc(sizeof(int) * project.K);

  fclose(file);
  return project;
}

int main() {
  Project project = parse_psplib("../sm_j10/PSP1.SCH");
  printf("%d %d\n", project.n, project.K);
}
