using CLSCA
using Test
using DataStructures
using NPZ

@testset "keyrank" begin

    # load /anfs/bigdisc/mgk25/scy27-keyrank/U-Os/Tables_bits/L01/table_0000.npy
    t = npzread("table_0000.npy")
    p = t[:,:,2]'
    v = UInt8.(t[:,:,1]')

    # some test keys with known rank
    kt = SortedDict{Int,Vector{UInt8}}(
        2^0  => [0xa1, 0x1b, 0x46, 0x37, 0x05, 0x40, 0x2a, 0xcb,
                 0x80, 0x99, 0xa8, 0x37, 0xd9, 0x34, 0xb1, 0x3b],
        2^1  => [0xa1, 0x1b, 0x46, 0x37, 0x05, 0x40, 0x2a, 0xcb,
                 0x80, 0x99, 0xb8, 0x37, 0xd9, 0x34, 0xb1, 0x3b],
        72   => [0xa1, 0x1b, 0x46, 0x37, 0x05, 0x40, 0x2b, 0xc3,
                 0x80, 0x99, 0xa8, 0x37, 0xd9, 0x34, 0xb1, 0x3b],
        2^20 => [0xa1, 0x1b, 0x46, 0x37, 0x01, 0x40, 0x2a, 0xda,
                 0x80, 0x9a, 0xb0, 0x33, 0xd9, 0x34, 0xb1, 0xbb],
        2^22 => [0xa1, 0x1b, 0x46, 0x37, 0x05, 0x40, 0x2a, 0xda,
                 0x80, 0x99, 0xa9, 0x3f, 0xdb, 0x30, 0xb1, 0x3b],
        2^24 => [0xa1, 0x1b, 0x46, 0x37, 0x01, 0x40, 0x3a, 0xda,
                 0x80, 0x99, 0xb8, 0x37, 0xd9, 0x30, 0xb1, 0x6d],
        2^26 => [0xa1, 0x1b, 0xc6, 0x37, 0x01, 0x41, 0x2a, 0xca,
                 0x80, 0x9b, 0xb0, 0x33, 0xdb, 0x30, 0xb1, 0x3b],
        2^28 => [0xa1, 0x1b, 0x46, 0x33, 0x05, 0x40, 0x2a, 0xe6,
                 0x80, 0x9b, 0xa4, 0x73, 0xd9, 0x31, 0xb1, 0x3b],
    )

    e = KeyEnumerator(p,v)

    for d = [1, 2, 72]
        (dt,prob) = depth(e, kt[d])
        @test dt == d
    end

    @time est, low, high = @time estimate_rank(p,v,kt[72])
    @test low <= 72 <= high

    for d = [1:127 ; 2 .^ (8:2:24)]
        if d <= 2^20
            if haskey(kt, d)
                @test kt[d] == nth(e, d)
            else
                kt[d] = nth(e, d)
            end
        end
        est, low, high = @time estimate_rank(p,v,kt[d])
        println("$low <= $est ~ $d <= $high")
        @test low <= d <= high && low <= est <= high
    end

    # test random keys with high rank
    for d = 1:1
        kr = rand(UInt8, 16);
        est, low, high = @time estimate_rank(p,v,kr)
        println("$low <= $est ~ $d <= $high")
        @test low <= est <= high
    end

end
