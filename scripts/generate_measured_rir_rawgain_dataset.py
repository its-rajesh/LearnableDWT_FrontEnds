#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, shutil
from pathlib import Path
import numpy as np
import scipy.signal as sp
import soundfile as sf

STEMS=("vocals","bass","drums")
EPS=1e-12

def sanitize(x): return np.nan_to_num(np.asarray(x,dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)
def rms(x): x=np.asarray(x,dtype=np.float64); return float(np.sqrt(np.mean(x*x)+EPS))
def peak(x): return float(np.max(np.abs(np.asarray(x)))+EPS)
def db10(a,b): return float(10*np.log10((a+EPS)/(b+EPS)))

def read_audio(path:Path, always_2d=False):
    x,sr=sf.read(str(path), dtype='float32', always_2d=always_2d)
    return sanitize(x), int(sr)

def resample(x, sr, target_sr, axis=0):
    if sr==target_sr: return sanitize(x)
    g=math.gcd(sr,target_sr)
    return sanitize(sp.resample_poly(x, target_sr//g, sr//g, axis=axis))

def highpass(x, sr, cutoff):
    if cutoff<=0: return sanitize(x)
    sos=sp.butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    axis=-1 if x.ndim==1 else 0
    return sanitize(sp.sosfiltfilt(sos, x, axis=axis))

def read_mono_resampled(path, target_sr):
    x,sr=read_audio(path, always_2d=False)
    if x.ndim==2: x=x.mean(axis=1)
    return resample(x, sr, target_sr, axis=0)

def convolve_crop(x,h,length):
    y=sp.fftconvolve(x.astype(np.float32), h.astype(np.float32), mode='full')
    if len(y)<length: y=np.pad(y,(0,length-len(y)))
    return sanitize(y[:length])

def load_rirs(rir_dir, target_sr):
    rirs={}; rows=[]
    for stem in STEMS:
        for mic in range(3):
            path=rir_dir/f'{stem}_mic{mic}_ir.wav'
            if not path.exists(): raise FileNotFoundError(path)
            h,sr=read_audio(path)
            if h.ndim==2: h=h.mean(axis=1)
            h=resample(h, sr, target_sr, axis=0)
            rirs[(stem,mic)]=h
            rows.append({'source':stem,'mic':mic,'path':str(path),'source_sr':sr,'target_sr':target_sr,'samples':len(h),'peak':peak(h),'rms':rms(h),'energy':float(np.sum(h.astype(np.float64)**2))})
    return rirs, rows

def load_noise(path, target_sr, length, gain, lowcut):
    if path is None: return None
    z,sr=read_audio(path, always_2d=True)
    # soundfile gives T,C. Preserve channels; if mono, replicate.
    if z.ndim==1: z=np.stack([z,z,z],axis=1)
    if z.shape[1]==1: z=np.repeat(z,3,axis=1)
    if z.shape[1]!=3 and z.shape[0]==3: z=z.T
    if z.shape[1]!=3: raise ValueError(f'Noise must be mono or 3-channel: {path}, shape={z.shape}')
    z=resample(z, sr, target_sr, axis=0)
    z=highpass(z, target_sr, lowcut)
    z=z.T.astype(np.float32) # C,T
    if z.shape[1]<length:
        reps=int(np.ceil(length/z.shape[1])); z=np.tile(z,(1,reps))
    return sanitize(gain*z[:,:length])

def song_names(old_dataset, split):
    return sorted([p.name for p in (old_dataset/split).iterdir() if p.is_dir()])

def load_sources(musdb_root, split, song, target_sr):
    d=musdb_root/split/song
    src={}; lengths=[]
    for stem in STEMS:
        x=read_mono_resampled(d/f'{stem}.wav', target_sr)
        src[stem]=x; lengths.append(len(x))
    L=min(lengths)
    return {k:sanitize(v[:L]) for k,v in src.items()}, L

def write_wav(path,x,sr):
    path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(str(path), sanitize(x), sr, subtype='FLOAT')

def generate_song(song, split, args, rirs, noise_path):
    sources,L=load_sources(args.musdb_root, split, song, args.target_sr)
    play_gains={}
    if args.play_rms and args.play_rms > 0:
        for stem in STEMS:
            g=float(args.play_rms/(rms(sources[stem])+EPS))
            sources[stem]=sanitize(sources[stem]*g)
            play_gains[stem]=g
    elif args.play_peak and args.play_peak > 0:
        for stem in STEMS:
            g=float(args.play_peak/(peak(sources[stem])+EPS))
            sources[stem]=sanitize(sources[stem]*g)
            play_gains[stem]=g
    else:
        play_gains={stem:1.0 for stem in STEMS}
    comps=np.zeros((3,3,L),dtype=np.float32)
    for si,stem in enumerate(STEMS):
        for mic in range(3): comps[si,mic]=convolve_crop(sources[stem], rirs[(stem,mic)], L)
    X=comps.sum(axis=0).astype(np.float32)
    noise=load_noise(noise_path,args.target_sr,L,args.noise_gain,args.lowcut) if noise_path else None
    if noise is not None: X=sanitize(X+noise)
    Y=np.stack([comps[i,i] for i in range(3)],axis=0).astype(np.float32)
    max_abs=max(peak(X),peak(Y)); scale=1.0
    if max_abs>args.peak_limit:
        scale=args.peak_limit/max_abs; X*=scale; Y*=scale; comps*=scale
        if noise is not None: noise*=scale
    if X.shape!=Y.shape: raise RuntimeError((X.shape,Y.shape))
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)): raise RuntimeError('NaN/Inf')
    out=args.out_dir/'synth_dataset'/split/song
    for i,stem in enumerate(STEMS):
        write_wav(out/'X'/f'{stem}.wav', X[i], args.target_sr)
        write_wav(out/'Y'/f'{stem}.wav', Y[i], args.target_sr)
    rows=[]
    for mic in range(3):
        target=comps[mic,mic].astype(np.float64)
        bleed=(comps[:,mic].sum(axis=0)-comps[mic,mic]).astype(np.float64)
        n=np.zeros_like(target) if noise is None else noise[mic].astype(np.float64)
        rows.append({
            'split':split,'song_id':song,'mic':mic,'sample_rate':args.target_sr,'length':L,'global_scale':scale,
            'target_rms':rms(target),'bleed_rms':rms(bleed),'noise_rms':rms(n),'mixture_rms':rms(X[mic]),
            'target_peak':peak(target),'bleed_peak':peak(bleed),'noise_peak':peak(n),'mixture_peak':peak(X[mic]),
            'target_to_bleed_db':db10(float(np.sum(target*target)), float(np.sum(bleed*bleed))),
            'target_to_noise_db':db10(float(np.sum(target*target)), float(np.sum(n*n))),
            'noise_added':noise is not None,
            'play_gain': play_gains[STEMS[mic]],
        })
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project_root',type=Path,default=Path('/home/rrame12/Desktop/Research/DWT_IR'))
    ap.add_argument('--musdb_root',type=Path,default=Path('/home/rrame12/Desktop/Datasets/musdb18hq'))
    ap.add_argument('--old_dataset',type=Path,default=Path('/home/rrame12/Desktop/Research/DWT_IR/measured_rir_synth_noise/synth_dataset'))
    ap.add_argument('--rir_dir',type=Path,default=Path('/home/rrame12/Desktop/Research/DWT_IR/measured_rir_rawgain/rir_wavs'))
    ap.add_argument('--out_dir',type=Path,default=Path('/home/rrame12/Desktop/Research/DWT_IR/measured_rir_rawgain_synth_smoke'))
    ap.add_argument('--noise_path',type=Path,default=Path('/home/rrame12/Desktop/Datasets/Re-recorded/sample_data/_session_room_measurement/session_room_noise.wav'))
    ap.add_argument('--noise_gain',type=float,default=1.0)
    ap.add_argument('--target_sr',type=int,default=22050)
    ap.add_argument('--lowcut',type=float,default=50.0)
    ap.add_argument('--peak_limit',type=float,default=0.99)
    ap.add_argument('--play_peak',type=float,default=0.12, help='Normalize each dry stem to this peak before convolution; <=0 disables')
    ap.add_argument('--play_rms',type=float,default=0.0, help='Normalize each dry stem to this RMS before convolution; overrides play_peak when >0')
    ap.add_argument('--max_train',type=int,default=5)
    ap.add_argument('--max_test',type=int,default=5)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    synth=args.out_dir/'synth_dataset'
    if synth.exists():
        if not args.overwrite: raise FileExistsError(f'{synth} exists; use --overwrite')
        shutil.rmtree(synth)
    rirs,rir_rows=load_rirs(args.rir_dir,args.target_sr)
    noise_path=args.noise_path if args.noise_path and args.noise_path.exists() else None
    all_rows=[]; counts={}
    for split,maxn in [('train',args.max_train),('test',args.max_test)]:
        songs=song_names(args.old_dataset,split)
        if maxn and maxn>0: songs=songs[:maxn]
        counts[split]=len(songs)
        print(f'[Generate] {split} {len(songs)} songs')
        for i,song in enumerate(songs,1):
            all_rows.extend(generate_song(song,split,args,rirs,noise_path))
            print(f'  {split} {i}/{len(songs)} {song}')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'metadata_components.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    with (args.out_dir/'rir_mapping.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rir_rows[0].keys())); w.writeheader(); w.writerows(rir_rows)
    manifest={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    manifest.update({'counts':counts,'noise_added':noise_path is not None,'noise_file':str(noise_path) if noise_path else ''})
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('[Done]', synth)
    print('metadata', args.out_dir/'metadata_components.csv')
if __name__=='__main__': main()
