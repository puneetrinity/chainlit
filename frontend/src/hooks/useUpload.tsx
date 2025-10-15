import { useCallback } from 'react';
import {
  DropzoneOptions,
  FileRejection,
  FileWithPath,
  useDropzone
} from 'react-dropzone';

import type { FileSpec } from 'client-types/';

interface useUploadProps {
  onError?: (error: string) => void;
  onResolved: (payloads: FileWithPath[]) => void;
  options?: DropzoneOptions;
  spec: FileSpec;
}

const useUpload = ({ onError, onResolved, options, spec }: useUploadProps) => {
  const onDrop: DropzoneOptions['onDrop'] = useCallback(
    (acceptedFiles: FileWithPath[], fileRejections: FileRejection[]) => {
      if (fileRejections.length > 0) {
        if (fileRejections[0].errors[0].code === 'file-too-large') {
          onError?.(`File is larger than ${spec.max_size_mb} MB`);
        } else {
          onError?.(fileRejections[0].errors[0].message);
        }
      }

      if (!acceptedFiles.length) return;
      return onResolved(acceptedFiles);
    },
    [spec]
  );

  // Normalize and sanitize accept so react-dropzone doesn't warn on invalid patterns like "*/*"
  let dzAccept: Record<string, string[]> | undefined = undefined;
  const accept = spec.accept;

  if (Array.isArray(accept)) {
    const filtered = accept.filter(
      (a) => typeof a === 'string' && a.trim() !== '*/*'
    );
    if (filtered.length) {
      dzAccept = {};
      filtered.forEach((a) => {
        dzAccept![a] = [];
      });
    }
  } else if (accept && typeof accept === 'object') {
    const entries = Object.entries(accept).filter(([mime]) => mime.trim() !== '*/*');
    if (entries.length) {
      dzAccept = Object.fromEntries(entries);
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: spec.max_files || undefined,
    // If accept is undefined (allow-all), omit to avoid browser/attr-accept warnings
    accept: dzAccept,
    maxSize: (spec.max_size_mb || 2) * 1000000,
    ...options
  });

  return { getInputProps, getRootProps, isDragActive };
};

export { useUpload };
