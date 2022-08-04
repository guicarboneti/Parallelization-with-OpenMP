#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/*
 * pRNG based on http://www.cs.wm.edu/~va/software/park/park.html
 */
#define MODULUS    2147483647
#define MULTIPLIER 48271
#define DEFAULT    123456789
#define NUM_THREADS 8

static long seed = DEFAULT;

double Random(void)
/* ----------------------------------------------------------------
 * Random returns a pseudo-random real number uniformly distributed 
 * between 0.0 and 1.0. 
 * ----------------------------------------------------------------
 */
{
  const long Q = MODULUS / MULTIPLIER;
  const long R = MODULUS % MULTIPLIER;
        long t;

  t = MULTIPLIER * (seed % Q) - R * (seed / Q);
  if (t > 0) 
    seed = t;
  else 
    seed = t + MODULUS;
  return ((double) seed / MODULUS);
}

/*
 * End of the pRNG algorithm
 */

typedef struct {
    double x, y, z;
    double mass;
    } Particle;
typedef struct {
    double xold, yold, zold;
    double fx, fy, fz;
    } ParticleV;

void InitParticles( Particle[], ParticleV [], int );
double ComputeForces( Particle [], Particle [], ParticleV [], int );
double ComputeNewPos( Particle [], ParticleV [], int, double);

int main()
{
    double start, end;
    start = omp_get_wtime();
    double time;
    Particle  * particles;   /* Particles */
    ParticleV * pv;          /* Particle velocity */
    int         npart, i, j;
    int         cnt;         /* number of times in loop */
    double      sim_t;       /* Simulation time */
    int tmp;
    tmp = fscanf(stdin,"%d\n",&npart);
    tmp = fscanf(stdin,"%d\n",&cnt);
/* Allocate memory for particles */
    particles = (Particle *) malloc(sizeof(Particle)*npart);
    pv = (ParticleV *) malloc(sizeof(ParticleV)*npart);
/* Generate the initial values */
    double startInit, endInit;
    startInit=omp_get_wtime();
    InitParticles( particles, pv, npart);
    endInit=omp_get_wtime();

    sim_t = 0.0;

    double startForces, startNew, endForces, endNew, timeFor, timeNew;
    timeFor=0.0;
    timeNew=0.0;
    double max_f;
    #pragma omp single
    for (i=0; i<cnt; i++) {
        /* Compute forces (2D only) */
        startForces = omp_get_wtime();
        max_f = ComputeForces( particles, particles, pv, npart );
        endForces = omp_get_wtime();
        timeFor += endForces - startForces;
        
        /* Once we have the forces, we compute the changes in position */
        startNew = omp_get_wtime();
        sim_t += ComputeNewPos( particles, pv, npart, max_f);
        endNew = omp_get_wtime();
        timeNew += endNew - startNew;
    }

    for (i=0; i<npart; i++)
        fprintf(stdout,"%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);

    end = omp_get_wtime();
    // printf("\nTEMPO MAIN: %f segundos\n", end - start);
    // printf("TEMPO INIT %f segundos\n", endInit - startInit);
    // printf("TEMPO FORCES %f segundos\n", timeFor);
    // printf("TEMPO NEW %f segundos\n", timeNew);
    return 0;
}

void InitParticles( Particle particles[], ParticleV pv[], int npart )
{
    int i;
    for (i=0; i<npart; i++) {
        particles[i].x	  = Random();
        particles[i].y	  = Random();
        particles[i].z	  = Random();
    }
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(auto)
    for (i=0; i<npart; i++) {
        particles[i].mass = 1.0;
        pv[i].xold	  = particles[i].x;
        pv[i].yold	  = particles[i].y;
        pv[i].zold	  = particles[i].z;
        pv[i].fx	  = 0;
        pv[i].fy	  = 0;
        pv[i].fz	  = 0;
    }
}

double ComputeForces( Particle myparticles[], Particle others[], ParticleV pv[], int npart )
{
    double max_f;
    int i;
    max_f = 0.0;
    int j;
    double xi, yi, mi, rx, ry, mj, r, fx, fy, rmin;
    #pragma omp parallel for private(xi, yi, rmin) reduction(+:fx,fy) num_threads(NUM_THREADS) schedule(auto)
    for (i=0; i<npart; i++) {
        rmin = 100.0;
        xi   = myparticles[i].x;
        yi   = myparticles[i].y;
        fx   = 0.0;
        fy   = 0.0;
        #pragma omp parallel for private(rx, ry, mj, r) reduction(-:fx,fy) num_threads(NUM_THREADS) schedule(auto)
        for (j=0; j<npart; j++) {
            rx = xi - others[j].x;
            ry = yi - others[j].y;
            mj = others[j].mass;
            r  = rx * rx + ry * ry;
            /* ignore overlap and same particle */
            if (r == 0.0) continue;
            if (r < rmin) rmin = r;
            r  = r * sqrt(r);
            fx -= mj * rx / r;
            fy -= mj * ry / r;
        }
        pv[i].fx += fx;
        pv[i].fy += fy;
        fx = sqrt(fx*fx + fy*fy)/rmin;
        if (fx > max_f) max_f = fx;
    }
    return max_f;
}

double ComputeNewPos( Particle particles[], ParticleV pv[], int npart, double max_f)
{
    int i;
    double a0, a1, a2;
    static double dt_old = 0.001, dt = 0.001;
    double dt_new;
    a0	 = 2.0 / (dt * (dt + dt_old));
    a2	 = 2.0 / (dt_old * (dt + dt_old));
    a1	 = -(a0 + a2);
    double xi, yi;
    #pragma omp parallel for private(xi, yi) num_threads(NUM_THREADS) schedule(auto)
    for (i=0; i<npart; i++) {
        xi	           = particles[i].x;
        yi	           = particles[i].y;
        particles[i].x = (pv[i].fx - a1 * xi - a2 * pv[i].xold) / a0;
        particles[i].y = (pv[i].fy - a1 * yi - a2 * pv[i].yold) / a0;
        pv[i].xold     = xi;
        pv[i].yold     = yi;
        pv[i].fx       = 0;
        pv[i].fy       = 0;
    }
    dt_new = 1.0/sqrt(max_f);
    /* Set a minimum: */
    if (dt_new < 1.0e-6) dt_new = 1.0e-6;
    /* Modify time step */
    if (dt_new < dt) {
        dt_old = dt;
        dt     = dt_new;
    }
    else if (dt_new > 4.0 * dt) {
        dt_old = dt;
        dt    *= 2.0;
    }
    return dt_old;
}

