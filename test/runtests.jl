using CLSCA
using Test
using Npy

# example data for key ranking
t = NpyArray("table_0000.npy")
p = t[:,:,2]'
v = UInt8.(t[:,:,1]')
k72 = UInt8[0xa1, 0x1b, 0x46, 0x37, 0x05, 0x40, 0x2b, 0xc3,
            0x80, 0x99, 0xa8, 0x37, 0xd9, 0x34, 0xb1, 0x3b]

@testset "keyrank" begin

    e = KeyEnumerator(p,v)

    @test let (d,prob) = depth(e, k72); d == 72 ; end

    @time est, low, high = @time estimate_rank(p,v,k72)
    @test low <= 72 <= high

    for d = [1:127 ; 2 .^ (7:24)]
        kt = nth(e, d)
        est, low, high = @time estimate_rank(p,v,kt)
        println("$low <= $est ~ $d <= $high")
        @test low <= d <= high && low <= est <= high
    end

    # test random keys with high
    for d = 1:1
        kt = rand(UInt8, 16);
        est, low, high = @time estimate_rank(p,v,kt)
        println("$low <= $est ~ $d <= $high")
        @test low <= est <= high
    end

end
