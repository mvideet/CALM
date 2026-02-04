from baukit import TraceDict
from ..model import *
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from entmax import sparsemax, entmax15, entmax_bisect, normmax_bisect, budget_bisect
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager that raises TimeoutException after specified seconds"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def load_model(model_name, cur_dataset, lora_path=None):

	if model_name == "qwen2-audio-instruct":
		from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
		device = "cuda" if torch.cuda.is_available() else "cpu"
		processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
		model = Qwen2AudioForConditionalGeneration.from_pretrained(
			"Qwen/Qwen2-Audio-7B-Instruct",
			torch_dtype=torch.float16 if device == "cuda" else torch.float32,
			device_map={"": device},
		)
		model.tie_weights()
		model.eval()
		model.requires_grad_(False)
		model_helper = Qwen2AudioHelper(model, processor, cur_dataset)
	elif model_name == "qwen2.5_omni":
		from transformers import (
			Qwen2_5OmniForConditionalGeneration,
			Qwen2_5OmniProcessor,
			Qwen2_5OmniThinkerForConditionalGeneration,
		)
		model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
			"Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto"
		)
		model.eval()
		model.requires_grad_(False)
		processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
		model_helper = Qwen2OmniHelper(model, processor, cur_dataset)
	else:
		raise ValueError(
			f"Unsupported model '{model_name}'. Use 'qwen2-audio-instruct' or 'qwen2.5_omni'."
		)

	return model_helper


def gather_last_attn_activations(inputs, model_helper):

	with TraceDict(
		model_helper.model,
		layers=model_helper.model_config['attn_hook_names'],
		retain_input=True,
		retain_output=True,
	) as td:
		result = model_helper.forward(inputs)
	return td, result


def split_activations_by_head(activations, model_config):

	if activations.dim() == 2:
		activations = activations.unsqueeze(1)

	new_shape = activations.size()[:-1] + (
		model_config['n_heads'],
		model_config['resid_dim'] // model_config['n_heads'],
	)
	activations = activations.view(*new_shape)
	return activations.to("cuda")


def get_last_mean_head_activations(entire_dataset, curr_item, model_helper, N_TRIALS = 50, shot=4, no_mean=False, split="train", audio_or_video="audio"):

	running_sum = None
	successful_trials = 0
	failed_trials = 0
	max_failed_trials = max(1, N_TRIALS // 2)  # Allow up to half the trials to fail
	
	# Print modality detection message only once per run (using function attribute)
	if not hasattr(get_last_mean_head_activations, '_modality_printed'):
		# Detect modality on first sample for debug output
		sample = curr_item if isinstance(curr_item, dict) else curr_item[0]
		
		# Quick check to determine modality for debug message
		try:
			result = model_helper.format_func(all_data=entire_dataset, cur_item=sample, num_shot=shot, model_helper=model_helper, split=split, audio_or_video=audio_or_video)
			if len(result) == 5:
				tqs, ans, audio_list, _, _ = result
				video_list = [None] * len(tqs)
			else:
				tqs, ans, video_list, audio_list, _, _ = result
			has_video = video_list and any(v is not None for v in video_list)
			has_insert_video = hasattr(model_helper, 'insert_video')
			detected_modality = "video" if (has_video and has_insert_video) else "audio"
			non_none_videos = sum(1 for v in video_list if v is not None) if video_list else 0
		except:
			detected_modality = "unknown"
			has_video = False
			has_insert_video = hasattr(model_helper, 'insert_video')
			non_none_videos = 0
		
		# Print nicely formatted debug message (only once)
		print("=" * 80)
		print("🔍 MODALITY DETECTION (PRWE)")
		print("=" * 80)
		print(f"  Requested modality:     {audio_or_video.upper()}")
		print(f"  Detected modality:     {detected_modality.upper()}")
		print(f"  Model supports video:  {'✓ YES' if has_insert_video else '✗ NO'}")
		print(f"  Video data available:  {'✓ YES' if has_video else '✗ NO'}")
		if has_video:
			print(f"  Video files found:      {non_none_videos}")
		print(f"  Final processing:      {detected_modality.upper()}-ONLY")
		print(f"  N_TRIALS:              {N_TRIALS}")
		print(f"  Shot:                  {shot}")
		print(f"  Split:                 {split}")
		print("=" * 80)
		
		# Mark as printed
		get_last_mean_head_activations._modality_printed = True
	
	for n in range(N_TRIALS):
		torch.cuda.empty_cache()
		# Safely extract sample from curr_item
		if isinstance(curr_item, dict):
			sample = curr_item
		elif isinstance(curr_item, list) and len(curr_item) > 0:
			sample = curr_item[0]
		else:
			# If curr_item is invalid, skip this item entirely
			print(f"⚠️  Invalid curr_item format: {type(curr_item)}, skipping item")
			break
		try:
			# Set timeout to 5 minutes (300 seconds) for the entire processing pipeline
			# Note: SIGALRM timeout may not work in all contexts (e.g., multithreading)
			# but it should work for single-threaded processing
			with timeout(300):
				result = model_helper.format_func(all_data=entire_dataset, cur_item=sample, num_shot=shot, model_helper=model_helper, split=split, audio_or_video=audio_or_video)
				# Handle both old format (5 values) and new format (6 values)
				if len(result) == 5:
					# Old format (audio mode): qs_list, ans_list, audio_list, cur_label, qid
					tqs, ans, audio_list, _, _ = result
					video_list = [None] * len(tqs)  # No video in audio mode
				else:  # 6 values: qs_list, ans_list, video_list, audio_list, cur_label, qid
					tqs, ans, video_list, audio_list, _, _ = result
				# Use insert_video if video_list is available AND model supports it, otherwise fall back to insert_audio
				has_video = video_list and any(v is not None for v in video_list)
				has_insert_video = hasattr(model_helper, 'insert_video')
				
				if has_video and has_insert_video:
					try:
						inputs = model_helper.insert_video(tqs, ans, video_list, audio_list=audio_list)
					except VideoProcessingTimeout as e:
						# Multiprocessing timeout - video decoder hung in C code
						print(f"⏱️  insert_video timed out ({e}); falling back to audio-only.")
						inputs = model_helper.insert_audio(tqs, ans, audio_list)
					except AttributeError as e:
						# PyAV <11 does not expose av.AVError
						print(f"⚠️  insert_video failed due to missing av.AVError ({e}); falling back to audio-only.")
						inputs = model_helper.insert_audio(tqs, ans, audio_list)
					except Exception as e:
						# Any other video processing failure: fall back to audio-only
						print(f"⚠️  insert_video failed with {type(e).__name__}: {e}; falling back to audio-only.")
						inputs = model_helper.insert_audio(tqs, ans, audio_list)
				else:
					# Fall back to audio-only processing
					inputs = model_helper.insert_audio(tqs, ans, audio_list)
				activations_td, result = gather_last_attn_activations(inputs, model_helper)
		except TimeoutException as e:
			# If a timeout occurs, log it and skip this trial
			print(f"⏱️  TIMEOUT in trial {n+1}/{N_TRIALS}: {e}")
			print(f"   Skipping this trial and continuing with next trial...")
			torch.cuda.empty_cache()
			failed_trials += 1
			if failed_trials >= max_failed_trials:
				print(f"⚠️  Too many failed trials ({failed_trials}/{N_TRIALS}), stopping early")
				break
			continue
		except Exception as e:
			# If an error occurs, log it and skip this trial
			import traceback
			error_type = type(e).__name__
			error_msg = str(e)
			print(f"❌ Error in trial {n+1}/{N_TRIALS} for sample: {error_type}: {error_msg}")
			if "list index out of range" in error_msg.lower():
				print(f"   This usually indicates empty audio_list or sequence. Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
			torch.cuda.empty_cache()
			failed_trials += 1
			# If this is the first trial and it fails, we should still try to continue
			# but if ALL trials fail, we'll raise at the end
			# If too many trials have failed, stop early
			if failed_trials >= max_failed_trials:
				print(f"⚠️  Too many failed trials ({failed_trials}/{N_TRIALS}), stopping early")
				break
			# Otherwise, skip this trial and continue with remaining trials
			continue
		
		# Immediately clear inputs to free video memory
		del inputs
		torch.cuda.empty_cache()

		# Process layer activations one at a time
		layer_head_tensors = []
		for name in model_helper.model_config["attn_hook_names"]:
			layer_tensor = split_activations_by_head(activations_td[name].input, model_helper.model_config)
			layer_head_tensors.append(layer_tensor)
			# Clear original activation to free memory
			del activations_td[name]
		
		# Clear activations_td and result completely
		del activations_td
		del result
		torch.cuda.empty_cache()
		
		# Stack and reshape
		stack_initial = torch.stack(layer_head_tensors, dim=0)
		del layer_head_tensors  # Free memory

		# Check for empty sequences (shouldn't happen, but safety check)
		if stack_initial.shape[2] == 0:  # seq dimension is 0
			raise RuntimeError(f"Empty sequence detected in activations. Shape: {stack_initial.shape}. This usually means all audio files failed to load and text processing also failed.")

		# Get last token activations and reshape
		cur_activation = stack_initial[:, -1, :, :, :]
		cur_activation = cur_activation.permute(0, 2, 1, 3)

		del stack_initial  # Free memory
		torch.cuda.empty_cache()

		if no_mean:
			if running_sum is None:
				running_sum = cur_activation
			else:
				running_sum = torch.cat([running_sum, cur_activation], dim=0)
		else:
			if running_sum is None:
				running_sum = cur_activation
			else:
				running_sum += cur_activation
			del cur_activation
		successful_trials += 1
		torch.cuda.empty_cache()

	if successful_trials == 0:
		raise RuntimeError(f"All {N_TRIALS} trials failed for item. Cannot compute activations.")
	
	if no_mean:
		return running_sum
	else:
		mean_activations = running_sum / successful_trials
		del running_sum
		return mean_activations


def get_class_activations(train_dataset, model, attn_heads, last_n_tokens: int = 1, audio_or_video="audio", n_trials=20):

	str_to_int = {}
	int_to_str = {}
	str_to_activation = {}
	str_to_count = {}
	save_act = {}

	for item in tqdm(train_dataset):
		try:
			mean_activations = get_last_mean_head_activations(train_dataset, item, model, N_TRIALS=n_trials, shot=0, audio_or_video=audio_or_video) 
			head_act = []
			for head in attn_heads:
				if isinstance(last_n_tokens, int) and last_n_tokens > 1:
					vec = mean_activations[head[0], head[1], -last_n_tokens:, :].mean(dim=0)
				else:
					vec = mean_activations[head[0], head[1], -1]
				head_act.append(vec)
			head_act = torch.stack(head_act)

			label = item['mapped_label']
			# Use lowercase key for consistent matching (handles case differences between train/test)
			label_key = label.lower() if isinstance(label, str) else label
			if label_key in str_to_activation:
				save_act[label_key] += [head_act]
				str_to_activation[label_key] += head_act
				str_to_count[label_key] += 1
			else:
				save_act[label_key] = [head_act]
				str_to_activation[label_key] = head_act
				int_label = len(str_to_activation.keys()) - 1
				str_to_int[label_key] = int_label
				int_to_str[int_label] = label_key  # Store lowercase for consistency
				str_to_count[label_key] = 1
		except (RuntimeError, OSError, Exception) as e:
			import warnings
			error_msg = str(e)
			error_type = type(e).__name__
			# Provide more helpful error messages for common issues
			if "AVError" in error_msg or "av" in error_msg.lower():
				warnings.warn(f"Failed to process item in get_class_activations (label: {item.get('mapped_label', 'unknown')}): Video processing error (likely PyAV version issue or missing video file). Error: {error_type}: {error_msg}. Skipping this item.")
			elif "No such file" in error_msg or "not found" in error_msg.lower():
				warnings.warn(f"Failed to process item in get_class_activations (label: {item.get('mapped_label', 'unknown')}): Missing video/audio file. Error: {error_type}: {error_msg}. Skipping this item.")
			else:
				warnings.warn(f"Failed to process item in get_class_activations (label: {item.get('mapped_label', 'unknown')}): {error_type}: {error_msg}. Skipping this item.")
			continue

	avg_activations = []
	for key, item in str_to_activation.items():
		save_act[key] = torch.stack(save_act[key], dim=0)
		avg_activations.append(torch.div(item, str_to_count[key]))
	avg_activations = torch.stack(avg_activations)
	return avg_activations, str_to_int, int_to_str


def get_query_activations(query_input, model_helper, common_heads, last_n_tokens: int = 1, audio_or_video="audio", n_trials=1):

	try:
		mean_activations = get_last_mean_head_activations(None, query_input, model_helper, N_TRIALS=n_trials, shot=0, audio_or_video=audio_or_video)
		head_act = []
		for head in common_heads:
			if isinstance(last_n_tokens, int) and last_n_tokens > 1:
				vec = mean_activations[head[0], head[1], -last_n_tokens:, :].mean(dim=0)
			else:
				vec = mean_activations[head[0], head[1], -1]
			head_act.append(vec)
		head_act = torch.stack(head_act)
		return head_act
	except (RuntimeError, OSError, Exception) as e:
		import warnings
		warnings.warn(f"Failed to process query in get_query_activations: {e}. Returning None.")
		return None


def prwe_prepare_cache(model, support_data, val_data, test_data=None, heads=None, last_n_tokens: int = 1, audio_or_video="audio", n_trials=20):
	# ============================================================================
	# ABLATION TOGGLE: Set to False to disable L2 normalization
	# ============================================================================
	USE_L2_NORMALIZATION = True  # Change to True to restore normalization
	# ============================================================================

	if heads is None:
		heads = list(model.all_heads)

	prototypes, str_to_int, int_to_str = get_class_activations(support_data, model, heads, last_n_tokens=last_n_tokens, audio_or_video=audio_or_video, n_trials=n_trials)
	C, K, D = prototypes.shape
	device = prototypes.device

	def _l2norm(x, dim=-1, eps=1e-8):
		denom = x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
		return x / denom

	cache_root = "/data/sls/u/urop/mvideet/sparse_audio/SAVs/cache"
	os.makedirs(cache_root, exist_ok=True)
	# Add _noL2norm suffix to cache directory when normalization is disabled
	cache_suffix = "_noL2norm" if not USE_L2_NORMALIZATION else ""
	subdir = f"prwe_C{C}_K{K}_D{D}_lastN{int(last_n_tokens)}{cache_suffix}"
	run_dir = os.path.join(cache_root, subdir)
	os.makedirs(run_dir, exist_ok=True)

	def _collect_qacts_and_labels(dataset, split_name: str):
		items = list(dataset)
		labels_idx = []
		original_indices = []
		split_dir = os.path.join(run_dir, split_name)
		os.makedirs(split_dir, exist_ok=True)
		valid_idx = 0
		for idx, it in enumerate(tqdm(items, desc="Precomputing query activations (cache)")):
			qa = get_query_activations([it], model, heads, last_n_tokens=last_n_tokens, audio_or_video=audio_or_video, n_trials=n_trials)
			if qa is None:
				import warnings
				warnings.warn(f"Skipping item {idx} in {split_name} due to failed query activation.")
				continue
			# Apply L2 normalization if enabled
			if USE_L2_NORMALIZATION:
				qa = _l2norm(qa, dim=1).to(dtype=prototypes.dtype, copy=False).cpu()
			else:
				qa = qa.to(dtype=prototypes.dtype, copy=False).cpu()
			torch.save(qa, os.path.join(split_dir, f"{split_name}_{valid_idx:06d}.pt"))
			y_str = it.get("mapped_label", it.get("label", ""))
			# Use lowercase for consistent matching (handles case differences between train/test)
			y_key = y_str.lower() if isinstance(y_str, str) else y_str
			labels_idx.append(str_to_int.get(y_key, -1))
			original_indices.append(idx)
			valid_idx += 1
		meta = {
			"type": "disk",
			"dir": split_dir,
			"pattern": f"{split_name}_%06d.pt",
			"count": valid_idx,
			"K": K,
			"D": D,
			"original_indices": original_indices,
		}
		return meta, labels_idx

	q_val_meta, val_labels_idx = _collect_qacts_and_labels(val_data, "val")
	q_test_meta, test_labels_idx = (None, None)
	if test_data is not None:
		q_test_meta, test_labels_idx = _collect_qacts_and_labels(test_data, "test")

	# Apply L2 normalization if enabled
	if USE_L2_NORMALIZATION:
		prot_n = _l2norm(prototypes, dim=2)
	else:
		prot_n = prototypes  # Use raw prototypes instead of normalized

	return {
		"prototypes_n": prot_n,
		"prototypes": prototypes,
		"heads": heads,
		"int_to_str": int_to_str,
		"str_to_int": str_to_int,
		"qacts_val_n": q_val_meta,
		"val_labels_idx": val_labels_idx,
		"qacts_test_n": q_test_meta,
		"test_labels_idx": test_labels_idx,
	}


def prwe_compute_posteriors_from_cache(cache, tau: float, split: str):

	prot_n = cache["prototypes_n"]
	if split == "val":
		q_src = cache["qacts_val_n"]
	elif split == "test":
		q_src = cache.get("qacts_test_n", None)
	else:
		raise ValueError("split must be 'val' or 'test'")

	if q_src is None:
		C = prot_n.shape[0]
		K = prot_n.shape[1]
		return torch.empty((0, K, C), dtype=prot_n.dtype, device=prot_n.device)

	device = prot_n.device
	dtype = prot_n.dtype

	if torch.is_tensor(q_src):
		if q_src.numel() == 0:
			C = prot_n.shape[0]
			K = prot_n.shape[1]
			return torch.empty((0, K, C), dtype=dtype, device=device)
		sims = torch.einsum("tjd,cjd->tjc", q_src.to(device=device, dtype=dtype), prot_n)
		logits = sims / float(tau)
		return torch.softmax(logits, dim=2)

	if isinstance(q_src, dict) and q_src.get("type") == "disk":
		count = int(q_src.get("count", 0))
		if count == 0:
			C = prot_n.shape[0]
			K = prot_n.shape[1]
			return torch.empty((0, K, C), dtype=dtype, device=device)
		split_dir = q_src["dir"]
		pattern = q_src["pattern"]
		chunk_size = 128
		out_chunks = []
		for start in range(0, count, chunk_size):
			end = min(start + chunk_size, count)
			batch = []
			for idx in range(start, end):
				path = os.path.join(split_dir, pattern % idx)
				batch.append(torch.load(path))
			q_batch = torch.stack(batch, dim=0).to(device=device, dtype=dtype, non_blocking=True)
			sims = torch.einsum("tjd,cjd->tjc", q_batch, prot_n)
			logits = sims / float(tau)
			out_chunks.append(torch.softmax(logits, dim=2).to(dtype=torch.float32).cpu())
			del q_batch, sims, logits
			torch.cuda.empty_cache()
		return torch.cat(out_chunks, dim=0)

	raise ValueError("Unsupported qacts source type in cache")


def prwe_compute_reliability(P_val: torch.Tensor, val_labels_idx, weight_scheme: str):

	T, K, C = P_val.shape
	device = P_val.device
	r = torch.zeros((K, C), dtype=P_val.dtype, device=device)
	counts = torch.zeros((C,), dtype=torch.long, device=device)

	idx_by_c = [[] for _ in range(C)]
	for t, y in enumerate(val_labels_idx):
		if 0 <= y < C:
			idx_by_c[y].append(t)

	for c in range(C):
		idx = idx_by_c[c]
		n_c = len(idx)
		counts[c] = n_c
		if n_c == 0:
			continue
		Pv = P_val[idx, :, :]

		if weight_scheme == "prob_softmax":
			r[:, c] = Pv[:, :, c].mean(dim=0)
		elif weight_scheme in ("margin_clamped", "margin_softmax"):
			# For each example in class c, for each head, find the two largest predicted class probabilities.
			# top_vals shape: (num_samples, num_heads, 2)
			# top_idx shape: (num_samples, num_heads, 2); top_idx[..., 0] is the class index of the largest probability, top_idx[..., 1] of the second largest
			# Handle case where there are fewer than 2 classes (can happen with binary classification if only one class exists)
			k = min(2, C)
			top_vals, top_idx = torch.topk(Pv, k=k, dim=2)
			p_jc = Pv[:, :, c]  # Model's confidence for the correct class c
			# For each example and head, if the top-1 class is c, use the top-2 value as the "next best" (max_other), else use the top-1 value.
			if k == 1:
				# Only one class exists - margin is just the confidence itself (no other class to compare against)
				max_other = torch.zeros_like(p_jc)
			else:
				max_other = torch.where(top_idx[:, :, 0] == c, top_vals[:, :, 1], top_vals[:, :, 0])
			# The margin is the difference between correct class confidence and the next best predicted class.
			margin = p_jc - max_other
			if weight_scheme == "margin_clamped":
				# Clamp negative margins to zero (so only positive margins count); negative margins become zero.
				margin = torch.clamp(margin, min=0.0)
			# For each head, average margin over all samples for this class c;
			# everything not in the top-2 is ignored—only the top-2 classes per sample matter in this calculation.
			r[:, c] = margin.mean(dim=0)
		elif weight_scheme == "brier_softmax":
			s2 = (Pv * Pv).sum(dim=2)
			r[:, c] = (2.0 * Pv[:, :, c] - s2).mean(dim=0)
		else:
			raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

	return r, counts


def prwe_apply_shrinkage(r: torch.Tensor, counts: torch.Tensor, alpha: float):
	if alpha <= 0.0:
		return r
	r_bar = r.mean()
	n_c = counts.to(r.dtype).unsqueeze(0)
	shrink = (n_c / (n_c + float(alpha)))
	comp = 1.0 - shrink
	return r * shrink + r_bar * comp


def prwe_build_weights_from_r(r_hat: torch.Tensor, *, weight_scheme: str, tau_w: float, top_k: int = None,alpha_entmax=1.0):
	K, C = r_hat.shape
	if weight_scheme in ("margin_softmax", "prob_softmax", "brier_softmax"):
		logits = r_hat / max(float(tau_w), 1e-8)
		
		if isinstance(top_k, int) and 0 < top_k < K:
			top_idx = torch.topk(r_hat, k=top_k, dim=0).indices
			mask = torch.zeros_like(logits, dtype=torch.bool)
			mask.scatter_(0, top_idx, True)
			min_val = torch.finfo(logits.dtype).min
			logits = logits.masked_fill(~mask, min_val)
		return torch.softmax(logits, dim=0)

	else:
		r = r_hat.clone()
		mask = None
		
		if isinstance(top_k, int) and 0 < top_k < K:
			top_idx = torch.topk(r, k=top_k, dim=0).indices
			mask = torch.zeros_like(r, dtype=torch.bool)
			mask.scatter_(0, top_idx, True)
			r = r.masked_fill(~mask, 0.0)

		tw = float(tau_w)
		if tw != 1.0:
			eps = torch.finfo(r.dtype).tiny
			gamma = 1.0 / max(tw, 1e-8)
			r = torch.pow(r + eps, gamma)

		sums = r.sum(dim=0, keepdim=True)
		w = torch.zeros_like(r)
		nonzero = (sums > 1e-12).squeeze(0)
		
		if nonzero.any():
			w[:, nonzero] = r[:, nonzero] / sums[:, nonzero]
			
		if (~nonzero).any():
			if mask is not None:
				allowed = mask[:, ~nonzero].to(r.dtype)
				allowed_sums = allowed.sum(dim=0, keepdim=True).clamp_min(1.0)
				w[:, ~nonzero] = allowed / allowed_sums
			else:
				w[:, ~nonzero] = 1.0 / float(K)
		return w


def prwe_eval_from_posteriors(P_test: torch.Tensor, w: torch.Tensor, *, test_labels_idx=None):

	if P_test is None or P_test.numel() == 0:
		return 0.0
	scores = (P_test * w.unsqueeze(0)).sum(dim=1)
	pred_idx = scores.argmax(dim=1)
	if test_labels_idx is None:
		return pred_idx
	correct = 0
	total = 0
	for i, y in enumerate(test_labels_idx):
		if 0 <= y < scores.shape[1]:
			correct += int(pred_idx[i].item() == y)
			total += 1
	return correct / max(total, 1)



