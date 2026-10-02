#pragma once

static std::string checkpointPath, resumePath, outputPrefix, rawPath;
static int checkpointEvery = 0;
static bool headless = false;
static std::ofstream statsFile;


inline void parseCommandLine(int argc, char** argv)
{
    for(int a=2;a<argc;++a) {
        std::string arg=argv[a];
        if(arg=="--headless") {headless=true;continue;}
        if(a+1>=argc) throw std::runtime_error("Missing value for "+arg);
        std::string value=argv[++a];
        if(arg=="--samples") {size_t used=0;int n=std::stoi(value,&used);if(n<=0||used!=value.size())throw std::runtime_error("Invalid sample count");scene->state.iterations=n;}
        else if(arg=="--output") outputPrefix=value;
        else if(arg=="--raw") rawPath=value;
        else if(arg=="--checkpoint") checkpointPath=value;
        else if(arg=="--resume") resumePath=value;
        else if(arg=="--checkpoint-every") {size_t used=0;checkpointEvery=std::stoi(value,&used);if(checkpointEvery<=0||used!=value.size())throw std::runtime_error("Invalid checkpoint interval");}
        else if(arg=="--stats") {statsFile.open(value);if(!statsFile)throw std::runtime_error("Cannot open stats file");}
        else throw std::runtime_error("Unknown argument: "+arg);
    }
    if(checkpointPath.empty()) checkpointPath=scene->state.imageName+".checkpoint";
    if(statsFile.is_open()) {statsFile<<"sample,generate_ms,intersect_ms,sort_ms,shade_ms,compact_ms,gather_ms,post_ms,active_after_bounce\n";pathtraceEnableStats(true);}

}
inline void restoreProgress()
{
    if(!resumePath.empty()) {
        auto start=std::chrono::steady_clock::now();iteration=loadCheckpoint(*scene,resumePath);
        std::cout<<"Loaded checkpoint in "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" ms\n";
    }
    if(iteration>static_cast<int>(renderState->iterations)) throw std::runtime_error("Sample target is below checkpoint count");
}
inline bool renderHeadless()
{
    if (!headless) return false;

        pathtraceInit(scene);
        auto start=std::chrono::steady_clock::now();
        int initial=iteration;
        while(iteration<static_cast<int>(renderState->iterations)) {
            pathtrace(nullptr,0,++iteration);recordStats();
            if(checkpointEvery>0&&iteration%checkpointEvery==0)saveProgress();
        }
        if(initial==iteration)pathtraceRefresh(nullptr,iteration);
        double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        saveImage();saveProgress();
        std::cout<<"Rendered "<<iteration-initial<<" new samples in "<<elapsed<<" ms ("<<iteration<<" total)\n";
        pathtraceFree();delete guiData;delete scene;scene=nullptr;return true;
}
void recordStats() {
    if(!statsFile.is_open())return;
    const auto& s=pathtraceLastStats();
    statsFile<<iteration<<','<<s.generateMs<<','<<s.intersectMs<<','<<s.sortMs<<','<<s.shadeMs<<','<<s.compactMs<<','<<s.gatherMs<<','<<s.postMs<<",\"";
    for(size_t i=0;i<s.activePaths.size();++i) {if(i)statsFile<<';';statsFile<<s.activePaths[i];}
    statsFile<<"\"\n";
}

void saveProgress() {
    if(iteration<=0)return;
    auto start=std::chrono::steady_clock::now();saveCheckpoint(*scene,iteration,checkpointPath);
    std::cout<<"Saved checkpoint "<<checkpointPath<<" in "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" ms\n";
}
