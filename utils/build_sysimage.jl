using PackageCompiler

create_sysimage(
    [:EulerAP];
    sysimage_path = "EulerAP.so",
    precompile_execution_file = "equations/relaxation_euler2d.jl"
)