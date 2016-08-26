//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <avilib.h>
#include <avimod.h>
#include <omp.h>

#include "define.c"
#include "kernel.c"

#ifdef GEM5_WORK
#include <stdint.h>
void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
#endif

#ifdef TIMING
#include <sys/time.h>
double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}
#endif

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

int main(int argc, char *argv []){

	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	// counters
	int i;
	int frames_processed;

	// parameters
	public_struct public;
	private_struct private[ALL_POINTS];

	//======================================================================================================================================================
	// 	FRAMES
	//======================================================================================================================================================

 	
	
	if(argc!=4){
		printf("ERROR: usage: heartwall <inputfile> <num of frames> <num of threads>\n");
		exit(1);
	}
	
	char* video_file_name;
	video_file_name = argv[1];
	
	avi_t* d_frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (d_frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	public.d_frames = d_frames;
	public.frames = AVI_video_frames(public.d_frames);
	public.frame_rows = AVI_video_height(public.d_frames);
	public.frame_cols = AVI_video_width(public.d_frames);
	public.frame_elem = public.frame_rows * public.frame_cols;
	public.frame_mem = sizeof(fp) * public.frame_elem;

	//======================================================================================================================================================
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================

	
	frames_processed = atoi(argv[2]);
	if(frames_processed<0 || frames_processed>public.frames){
		printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, public.frames);
		return 0;
	}
	
	int omp_num_threads;
	omp_num_threads = atoi(argv[3]);
	if (omp_num_threads <=0){
	   printf ("num of threads must be a positive integer");
	   return 0;
	}
	
	printf("num of threads: %d\n", omp_num_threads);
	
	//======================================================================================================================================================
	//	INPUTS
	//======================================================================================================================================================

	//====================================================================================================
	//	ENDO POINTS
	//====================================================================================================

	public.endoPoints = ENDO_POINTS;
	public.d_endo_mem = sizeof(int) * public.endoPoints;
	public.d_endoRow = (int *)malloc(public.d_endo_mem);
	public.d_endoRow[ 0] = 369;
	public.d_endoRow[ 1] = 400;
	public.d_endoRow[ 2] = 429;
	public.d_endoRow[ 3] = 452;
	public.d_endoRow[ 4] = 476;
	public.d_endoRow[ 5] = 486;
	public.d_endoRow[ 6] = 479;
	public.d_endoRow[ 7] = 458;
	public.d_endoRow[ 8] = 433;
	public.d_endoRow[ 9] = 404;
	public.d_endoRow[10] = 374;
	public.d_endoRow[11] = 346;
	public.d_endoRow[12] = 318;
	public.d_endoRow[13] = 294;
	public.d_endoRow[14] = 277;
	public.d_endoRow[15] = 269;
	public.d_endoRow[16] = 275;
	public.d_endoRow[17] = 287;
	public.d_endoRow[18] = 311;
	public.d_endoRow[19] = 339;
	public.d_endoCol = (int *)malloc(public.d_endo_mem);
	public.d_endoCol[ 0] = 408;
	public.d_endoCol[ 1] = 406;
	public.d_endoCol[ 2] = 397;
	public.d_endoCol[ 3] = 383;
	public.d_endoCol[ 4] = 354;
	public.d_endoCol[ 5] = 322;
	public.d_endoCol[ 6] = 294;
	public.d_endoCol[ 7] = 270;
	public.d_endoCol[ 8] = 250;
	public.d_endoCol[ 9] = 237;
	public.d_endoCol[10] = 235;
	public.d_endoCol[11] = 241;
	public.d_endoCol[12] = 254;
	public.d_endoCol[13] = 273;
	public.d_endoCol[14] = 300;
	public.d_endoCol[15] = 328;
	public.d_endoCol[16] = 356;
	public.d_endoCol[17] = 383;
	public.d_endoCol[18] = 401;
	public.d_endoCol[19] = 411;
	public.d_tEndoRowLoc = (int *)malloc(public.d_endo_mem * public.frames);
	public.d_tEndoColLoc = (int *)malloc(public.d_endo_mem * public.frames);

	//====================================================================================================
	//	EPI POINTS
	//====================================================================================================

	public.epiPoints = EPI_POINTS;
	public.d_epi_mem = sizeof(int) * public.epiPoints;
	public.d_epiRow = (int *)malloc(public.d_epi_mem);
	public.d_epiRow[ 0] = 390;
	public.d_epiRow[ 1] = 419;
	public.d_epiRow[ 2] = 448;
	public.d_epiRow[ 3] = 474;
	public.d_epiRow[ 4] = 501;
	public.d_epiRow[ 5] = 519;
	public.d_epiRow[ 6] = 535;
	public.d_epiRow[ 7] = 542;
	public.d_epiRow[ 8] = 543;
	public.d_epiRow[ 9] = 538;
	public.d_epiRow[10] = 528;
	public.d_epiRow[11] = 511;
	public.d_epiRow[12] = 491;
	public.d_epiRow[13] = 466;
	public.d_epiRow[14] = 438;
	public.d_epiRow[15] = 406;
	public.d_epiRow[16] = 376;
	public.d_epiRow[17] = 347;
	public.d_epiRow[18] = 318;
	public.d_epiRow[19] = 291;
	public.d_epiRow[20] = 275;
	public.d_epiRow[21] = 259;
	public.d_epiRow[22] = 256;
	public.d_epiRow[23] = 252;
	public.d_epiRow[24] = 252;
	public.d_epiRow[25] = 257;
	public.d_epiRow[26] = 266;
	public.d_epiRow[27] = 283;
	public.d_epiRow[28] = 305;
	public.d_epiRow[29] = 331;
	public.d_epiRow[30] = 360;
	public.d_epiCol = (int *)malloc(public.d_epi_mem);
	public.d_epiCol[ 0] = 457;
	public.d_epiCol[ 1] = 454;
	public.d_epiCol[ 2] = 446;
	public.d_epiCol[ 3] = 431;
	public.d_epiCol[ 4] = 411;
	public.d_epiCol[ 5] = 388;
	public.d_epiCol[ 6] = 361;
	public.d_epiCol[ 7] = 331;
	public.d_epiCol[ 8] = 301;
	public.d_epiCol[ 9] = 273;
	public.d_epiCol[10] = 243;
	public.d_epiCol[11] = 218;
	public.d_epiCol[12] = 196;
	public.d_epiCol[13] = 178;
	public.d_epiCol[14] = 166;
	public.d_epiCol[15] = 157;
	public.d_epiCol[16] = 155;
	public.d_epiCol[17] = 165;
	public.d_epiCol[18] = 177;
	public.d_epiCol[19] = 197;
	public.d_epiCol[20] = 218;
	public.d_epiCol[21] = 248;
	public.d_epiCol[22] = 276;
	public.d_epiCol[23] = 304;
	public.d_epiCol[24] = 333;
	public.d_epiCol[25] = 361;
	public.d_epiCol[26] = 391;
	public.d_epiCol[27] = 415;
	public.d_epiCol[28] = 434;
	public.d_epiCol[29] = 448;
	public.d_epiCol[30] = 455;
	public.d_tEpiRowLoc = (int *)malloc(public.d_epi_mem * public.frames);
	public.d_tEpiColLoc = (int *)malloc(public.d_epi_mem * public.frames);

	//====================================================================================================
	//	ALL POINTS
	//====================================================================================================

	public.allPoints = ALL_POINTS;

	//======================================================================================================================================================
	//	CONSTANTS
	//======================================================================================================================================================

	public.tSize = 25;
	public.sSize = 40;
	public.maxMove = 10;
	public.alpha = 0.87;

	//======================================================================================================================================================
	//	SUMS
	//======================================================================================================================================================

	for(i=0; i<public.allPoints; i++){
		private[i].in_partial_sum = (fp *)malloc(sizeof(fp) * 2*public.tSize+1);
		private[i].in_sqr_partial_sum = (fp *)malloc(sizeof(fp) * 2*public.tSize+1);
		private[i].par_max_val = (fp *)malloc(sizeof(fp) * (2*public.tSize+2*public.sSize+1));
		private[i].par_max_coo = (int *)malloc(sizeof(int) * (2*public.tSize+2*public.sSize+1));
	}

	//======================================================================================================================================================
	// 	INPUT 2 (SAMPLE AROUND POINT)
	//======================================================================================================================================================

	public.in2_rows = 2 * public.sSize + 1;
	public.in2_cols = 2 * public.sSize + 1;
	public.in2_elem = public.in2_rows * public.in2_cols;
	public.in2_mem = sizeof(fp) * public.in2_elem;

	for(i=0; i<public.allPoints; i++){
		private[i].d_in2 = (fp *)malloc(public.in2_mem);
		private[i].d_in2_sqr = (fp *)malloc(public.in2_mem);
	}

	//======================================================================================================================================================
	// 	INPUT (POINT TEMPLATE)
	//======================================================================================================================================================

	public.in_mod_rows = public.tSize+1+public.tSize;
	public.in_mod_cols = public.in_mod_rows;
	public.in_mod_elem = public.in_mod_rows * public.in_mod_cols;
	public.in_mod_mem = sizeof(fp) * public.in_mod_elem;

	for(i=0; i<public.allPoints; i++){
		private[i].d_in_mod = (fp *)malloc(public.in_mod_mem);
		private[i].d_in_sqr = (fp *)malloc(public.in_mod_mem);
	}

	//======================================================================================================================================================
	// 	ARRAY OF TEMPLATES FOR ALL POINTS
	//======================================================================================================================================================

	public.d_endoT = (fp *)malloc(public.in_mod_mem * public.endoPoints);
	public.d_epiT = (fp *)malloc(public.in_mod_mem * public.epiPoints);

	//======================================================================================================================================================
	// 	SETUP private POINTERS TO ROWS, COLS  AND TEMPLATE
	//======================================================================================================================================================

	for(i=0; i<public.endoPoints; i++){
		private[i].point_no = i;
		private[i].in_pointer = private[i].point_no * public.in_mod_elem;
		private[i].d_Row = public.d_endoRow;												// original row coordinates
		private[i].d_Col = public.d_endoCol;													// original col coordinates
		private[i].d_tRowLoc = public.d_tEndoRowLoc;									// updated row coordinates
		private[i].d_tColLoc = public.d_tEndoColLoc;										// updated row coordinates
		private[i].d_T = public.d_endoT;														// templates
	}

	for(i=public.endoPoints; i<public.allPoints; i++){
		private[i].point_no = i-public.endoPoints;
		private[i].in_pointer = private[i].point_no * public.in_mod_elem;
		private[i].d_Row = public.d_epiRow;
		private[i].d_Col = public.d_epiCol;
		private[i].d_tRowLoc = public.d_tEpiRowLoc;
		private[i].d_tColLoc = public.d_tEpiColLoc;
		private[i].d_T = public.d_epiT;
	}

	//======================================================================================================================================================
	// 	CONVOLUTION
	//======================================================================================================================================================

	public.ioffset = 0;
	public.joffset = 0;
	public.conv_rows = public.in_mod_rows + public.in2_rows - 1;												// number of rows in I
	public.conv_cols = public.in_mod_cols + public.in2_cols - 1;												// number of columns in I
	public.conv_elem = public.conv_rows * public.conv_cols;												// number of elements
	public.conv_mem = sizeof(fp) * public.conv_elem;

	for(i=0; i<public.allPoints; i++){
		private[i].d_conv = (fp *)malloc(public.conv_mem);
	}

	//======================================================================================================================================================
	// 	CUMULATIVE SUM
	//======================================================================================================================================================

	//====================================================================================================
	//	PAD ARRAY
	//====================================================================================================
	//====================================================================================================
	//	VERTICAL CUMULATIVE SUM
	//====================================================================================================

	public.in2_pad_add_rows = public.in_mod_rows;
	public.in2_pad_add_cols = public.in_mod_cols;
	public.in2_pad_rows = public.in2_rows + 2*public.in2_pad_add_rows;
	public.in2_pad_cols = public.in2_cols + 2*public.in2_pad_add_cols;
	public.in2_pad_elem = public.in2_pad_rows * public.in2_pad_cols;
	public.in2_pad_mem = sizeof(fp) * public.in2_pad_elem;

	for(i=0; i<public.allPoints; i++){
		private[i].d_in2_pad = (fp *)malloc(public.in2_pad_mem);
	}

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION
	//====================================================================================================
	//====================================================================================================
	//	HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	public.in2_pad_cumv_sel_rowlow = 1 + public.in_mod_rows;													// (1 to n+1)
	public.in2_pad_cumv_sel_rowhig = public.in2_pad_rows - 1;
	public.in2_pad_cumv_sel_collow = 1;
	public.in2_pad_cumv_sel_colhig = public.in2_pad_cols;
	public.in2_pad_cumv_sel2_rowlow = 1;
	public.in2_pad_cumv_sel2_rowhig = public.in2_pad_rows - public.in_mod_rows - 1;
	public.in2_pad_cumv_sel2_collow = 1;
	public.in2_pad_cumv_sel2_colhig = public.in2_pad_cols;
	public.in2_sub_rows = public.in2_pad_cumv_sel_rowhig - public.in2_pad_cumv_sel_rowlow + 1;
	public.in2_sub_cols = public.in2_pad_cumv_sel_colhig - public.in2_pad_cumv_sel_collow + 1;
	public.in2_sub_elem = public.in2_sub_rows * public.in2_sub_cols;
	public.in2_sub_mem = sizeof(fp) * public.in2_sub_elem;

	for(i=0; i<public.allPoints; i++){
		private[i].d_in2_sub = (fp *)malloc(public.in2_sub_mem);
	}

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION, SQUARE, NUMERATOR
	//====================================================================================================

	public.in2_sub_cumh_sel_rowlow = 1;
	public.in2_sub_cumh_sel_rowhig = public.in2_sub_rows;
	public.in2_sub_cumh_sel_collow = 1 + public.in_mod_cols;
	public.in2_sub_cumh_sel_colhig = public.in2_sub_cols - 1;
	public.in2_sub_cumh_sel2_rowlow = 1;
	public.in2_sub_cumh_sel2_rowhig = public.in2_sub_rows;
	public.in2_sub_cumh_sel2_collow = 1;
	public.in2_sub_cumh_sel2_colhig = public.in2_sub_cols - public.in_mod_cols - 1;
	public.in2_sub2_sqr_rows = public.in2_sub_cumh_sel_rowhig - public.in2_sub_cumh_sel_rowlow + 1;
	public.in2_sub2_sqr_cols = public.in2_sub_cumh_sel_colhig - public.in2_sub_cumh_sel_collow + 1;
	public.in2_sub2_sqr_elem = public.in2_sub2_sqr_rows * public.in2_sub2_sqr_cols;
	public.in2_sub2_sqr_mem = sizeof(fp) * public.in2_sub2_sqr_elem;

	for(i=0; i<public.allPoints; i++){
		private[i].d_in2_sub2_sqr = (fp *)malloc(public.in2_sub2_sqr_mem);
	}

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	PAD ARRAY
	//====================================================================================================
	//====================================================================================================
	//	VERTICAL CUMULATIVE SUM
	//====================================================================================================

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION
	//====================================================================================================
	//====================================================================================================
	//	HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	//====================================================================================================
	//	SELECTION, SELECTION 2, SUBTRACTION, DIFFERENTIAL LOCAL SUM, DENOMINATOR A, DENOMINATOR, CORRELATION
	//====================================================================================================

	//======================================================================================================================================================
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================

	public.tMask_rows = public.in_mod_rows + (public.sSize+1+public.sSize) - 1;
	public.tMask_cols = public.tMask_rows;
	public.tMask_elem = public.tMask_rows * public.tMask_cols;
	public.tMask_mem = sizeof(fp) * public.tMask_elem;

	for(i=0; i<public.allPoints; i++){
		private[i].d_tMask = (fp *)malloc(public.tMask_mem);
	}

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	public.mask_rows = public.maxMove;
	public.mask_cols = public.mask_rows;
	public.mask_elem = public.mask_rows * public.mask_cols;
	public.mask_mem = sizeof(fp) * public.mask_elem;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	public.mask_conv_rows = public.tMask_rows;												// number of rows in I
	public.mask_conv_cols = public.tMask_cols;												// number of columns in I
	public.mask_conv_elem = public.mask_conv_rows * public.mask_conv_cols;												// number of elements
	public.mask_conv_mem = sizeof(fp) * public.mask_conv_elem;
	public.mask_conv_ioffset = (public.mask_rows-1)/2;
	if((public.mask_rows-1) % 2 > 0.5){
		public.mask_conv_ioffset = public.mask_conv_ioffset + 1;
	}
	public.mask_conv_joffset = (public.mask_cols-1)/2;
	if((public.mask_cols-1) % 2 > 0.5){
		public.mask_conv_joffset = public.mask_conv_joffset + 1;
	}

	for(i=0; i<public.allPoints; i++){
		private[i].d_mask_conv = (fp *)malloc(public.mask_conv_mem);
	}

	//======================================================================================================================================================
	//	PRINT FRAME PROGRESS START
	//======================================================================================================================================================

	printf("frame progress: ");
	fflush(NULL);

	//======================================================================================================================================================
	//	KERNEL
	//======================================================================================================================================================

#ifdef GEM5_WORK
	m5_work_begin(0, 0);
	m5_dumpreset_stats(0, 0);
#endif

#ifdef TIMING
    double start_time = gettime();
#endif

	for(public.frame_no=0; public.frame_no<frames_processed; public.frame_no++){

	//====================================================================================================
	//	GETTING FRAME
	//====================================================================================================

		// Extract a cropped version of the first frame from the video file
		public.d_frame = get_frame(public.d_frames,				// pointer to video file
													public.frame_no,				// number of frame that needs to be returned
													0,										// cropped?
													0,										// scaled?
													1);									// converted

	//====================================================================================================
	//	PROCESSING
	//====================================================================================================

		omp_set_num_threads(omp_num_threads);
		

		#pragma omp parallel for
		for(i=0; i<public.allPoints; i++){
			kernel(	public,
						private[i]);
		}

	//====================================================================================================
	//	FREE MEMORY FOR FRAME
	//====================================================================================================

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		free(public.d_frame);

	//====================================================================================================
	//	PRINT FRAME PROGRESS
	//====================================================================================================

		printf("%d ", public.frame_no);
		fflush(NULL);

	}

	//======================================================================================================================================================
	//	PRINT FRAME PROGRESS END
	//======================================================================================================================================================

	printf("\n");
	fflush(NULL);

#ifdef TIMING
    double end_time = gettime();
    printf("ROI Runtime: %f\n", end_time - start_time);
#endif

#ifdef GEM5_WORK
    m5_dumpreset_stats(0, 0);
    m5_work_end(0, 0);
#endif

#ifdef BENCH_PRINT
	int f;
	for(f=0; f<frames_processed; f++){
		printf("frame:%d\n", f);
		for(i=f; i<public.endoPoints*public.frames; i+=public.frames){
			printf("%d\t", public.d_tEndoRowLoc[i]);
		}
			printf("\n");
		for(i=f; i<public.endoPoints*public.frames; i+=public.frames){
			printf("%d\t", public.d_tEndoColLoc[i]);
		}
			printf("\n");
		for(i=f; i<public.epiPoints*public.frames; i+=public.frames){
			printf("%d\t", public.d_tEpiRowLoc[i]);
		}
			printf("\n");
		for(i=f; i<public.epiPoints*public.frames; i+=public.frames){
			printf("%d\t", public.d_tEpiColLoc[i]);
		}
		printf("\n");
	}
#endif

	//======================================================================================================================================================
	//	DEALLOCATION
	//======================================================================================================================================================

	//====================================================================================================
	//	COMMON
	//====================================================================================================

	free(public.d_endoRow);
	free(public.d_endoCol);
	free(public.d_tEndoRowLoc);
	free(public.d_tEndoColLoc);
	free(public.d_endoT);

	free(public.d_epiRow);
	free(public.d_epiCol);
	free(public.d_tEpiRowLoc);
	free(public.d_tEpiColLoc);
	free(public.d_epiT);

	//====================================================================================================
	//	POINTERS
	//====================================================================================================

	for(i=0; i<public.allPoints; i++){
		free(private[i].in_partial_sum);
		free(private[i].in_sqr_partial_sum);
		free(private[i].par_max_val);
		free(private[i].par_max_coo);

		free(private[i].d_in2);
		free(private[i].d_in2_sqr);

		free(private[i].d_in_mod);
		free(private[i].d_in_sqr);

		free(private[i].d_conv);

		free(private[i].d_in2_pad);

		free(private[i].d_in2_sub);

		free(private[i].d_in2_sub2_sqr);

		free(private[i].d_tMask);
		free(private[i].d_mask_conv);
	}

}

//========================================================================================================================================================================================================
//========================================================================================================================================================================================================
//	END OF FILE
//========================================================================================================================================================================================================
//========================================================================================================================================================================================================
